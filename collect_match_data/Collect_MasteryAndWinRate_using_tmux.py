from riotwatcher import LolWatcher, ApiError  #
from pprint import pprint
from tqdm import tqdm  #
import datetime
from contextlib import contextmanager


import pandas as pd  #
import sys
import os

import time

from collections import defaultdict, deque
import numpy as np
import json
from operator import attrgetter

import random

KST = datetime.timezone(datetime.timedelta(hours=9))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    # print(f"[{name}] done in {time.time() - t0:.3f} s")


data_path = "../data"


riot_token = (
    "RGAPI-b7bcb5cc-317c-459e-850a-62c4685dca0a"  # 토큰 넣어주세요. 수집 데이터가 많을 때는 재발급하고 넣어주세요
)

# lol_watcher = LolWatcher(riot_token)
my_region = "kr"

# me = lol_watcher.summoner.by_name(my_region, "hide on bush")
# pprint(me)

# tiers = ['DIAMOND', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON']
# divisions = ['I', 'II', 'III', 'IV']

targetTier = "PLATINUM"  # 티어 넣어주세요
targetDivision = "III"  # 구간 넣어주세요

queue = "RANKED_SOLO_5x5"
queueId = 420

# 목표 : 티어 + 구간 당 250개 수집

goal = 10
maxMatchData = 500


class Riot:
    def __init__(self, token):
        self.token = token
        self.watcher = LolWatcher(token)
        self.page = 1

    def chooseNextTarget(self, tier, division):
        while True:
            summonerList = self.getResponse(
                "league.entries", my_region, queue, tier, division, self.page
            )
            if len(summonerList) == 0:
                self.page = 1
            else:
                self.page += 1
                break

        i = random.randrange(len(summonerList))
        return summonerList[i]["summonerId"]

    def printError(self, response):
        print()
        pprint(response.json())
        cmd = input("\033[31m" + "(Restart: AnyKey, Token: t) -> " + "\033[0m")
        if cmd == "t":
            self.token = input("\033[31m" + "    input token -> " + "\033[0m")
            self.watcher = LolWatcher(self.token)
        print()

    def getResponse(self, callName, *args):
        while True:
            try:
                reponse = attrgetter(callName)(self.watcher)(*args)
                break
            except ApiError as err:
                if err.response.status_code == 429:
                    time.sleep(5)
                elif err.response.status_code == 503:
                    pass

                else:
                    self.printError(err.response)

        return reponse

    def getSummoner(self, summonerName):  #
        return self.getResponse("summoner.by_name", my_region, summonerName)

    def getMatchId_one(self, summonerId):  #
        puuid = self.getResponse("summoner.by_id", my_region, summonerId)["puuid"]
        matchId = self.getResponse(
            "match.matchlist_by_puuid", "asia", puuid, 0, 1, queueId
        )

        return matchId[0]

    def getMatchId_many(self, summonerId, limit=1000000):  #
        puuid = self.getResponse("summoner.by_id", my_region, summonerId)["puuid"]

        startIndex = 0
        count = 100

        matchIdList = []
        while True:
            matchIDs = self.getResponse(
                "match.matchlist_by_puuid", "asia", puuid, startIndex, count, queueId
            )
            matchIdList.extend(matchIDs)
            if len(matchIDs) == 0 or len(matchIdList) >= limit:
                break
            startIndex += count

        return matchIdList

    def getMatchData_one(self, matchId):  #
        matchData = self.getResponse("match.by_id", "asia", matchId)
        return self.compactMatchData(matchData)

    def getChampionMastery(self, summonerId):  #
        return self.getResponse("champion_mastery.by_summoner", my_region, summonerId)

    def compactMatchData(self, matchData):
        cMatchData = {
            "matchId": matchData["metadata"]["matchId"],
            "gameCreation": matchData["info"]["gameCreation"],
            "gameDuration": matchData["info"]["gameDuration"],
            "gameVersion": matchData["info"]["gameVersion"],
            "participants": {},
        }

        for participant in matchData["info"]["participants"]:
            cMatchData["participants"][participant["summonerId"]] = {
                "championId": participant["championId"],
                #'summonerId': participant['summonerId'],
                "teamId": participant["teamId"],
                "teamPosition": participant["teamPosition"],
                "win": participant["win"],
            }

        return cMatchData


# 한번만 실행해주세요
mastery_dict = {}

summoner_done = set()
matchId_done = set()

targetSummonerQueue = deque()

matchIdToData = {}  # matchId -> compactMatchData
summonerIdToData = {}  # summonerId -> championId -> set(matchId, matchId, ....)

data_dict = {}


riot_token = input("\033[32m" + "Enter riot API key to continue: " + "\033[0m")
riot = Riot(riot_token)


def getChampionMastery(summonerId, championId):
    if summonerId not in mastery_dict:
        # 수집
        mastery_dict[summonerId] = defaultdict(int)

        # while True:
        #     try:
        #         raw_mastery = lol_watcher.champion_mastery.by_summoner(
        #             my_region, summonerId
        #         )
        #     except ApiError as err:
        #         printError(err)
        #     else:
        #         break

        raw_mastery = riot.getChampionMastery(summonerId)

        for masteryInfo in raw_mastery:
            mastery_dict[summonerId][masteryInfo["championId"]] = masteryInfo[
                "championPoints"
            ]

    return mastery_dict[summonerId][championId]


def getWinRate(matchIdList, summonerId):
    if len(matchIdList) == 0:
        return 0.0
    win = 0
    for matchId in matchIdList:
        if matchIdToData[matchId]["participants"][summonerId]["win"]:
            win += 1
    return win / len(matchIdList)


teamAndLines = [
    "100_TOP",
    "100_JUNGLE",
    "100_MIDDLE",
    "100_BOTTOM",
    "100_UTILITY",
    "200_TOP",
    "200_JUNGLE",
    "200_MIDDLE",
    "200_BOTTOM",
    "200_UTILITY",
]

pbar = tqdm(range(goal - len(data_dict)))
for _ in pbar:

    # choose target
    pbar.set_description("Choose target...")
    while True:
        if len(targetSummonerQueue) == 0:
            summonerId = riot.chooseNextTarget(targetTier, targetDivision)
            targetSummonerQueue.append(summonerId)

        targetSummonerId = targetSummonerQueue.popleft()

        if targetSummonerId in summoner_done:
            continue

        targetMatchId = riot.getMatchId_one(targetSummonerId)

        if targetMatchId not in matchId_done:
            break

    targetMatchData = riot.getMatchData_one(targetMatchId)
    matchIdToData[targetMatchId] = targetMatchData

    # 10명의 매치데이터 수집 (챔피언 승률)
    done = True
    temp_dict = {}
    for i, (summonerId, participant) in enumerate(
        targetMatchData["participants"].items()
    ):
        pbar.set_description(f"10명 매치데이터 수집 [{i+1}/10]")

        championId = participant["championId"]
        teamId = participant["teamId"]
        position = participant["teamPosition"]

        teamAndLine = "_".join([str(teamId), position])

        if teamAndLine not in teamAndLines:
            done = False
            break

        # collect match IDs
        pbar.set_description(f"Collect match IDs... [{i+1}/10]")
        matchIDs = riot.getMatchId_many(summonerId, limit=maxMatchData)

        # 플레이했던 매치 데이터, Id를 소환사ID -> 챔피언ID -> 매치아이디 딕셔너리로 구성
        pbar.set_description(f"Collect match Datas... [{i+1}/10]")
        for matchId in matchIDs[:maxMatchData]:
            # 수집해야 하는 경우
            if matchId not in matchIdToData:
                matchIdToData[matchId] = riot.getMatchData_one(matchId)

            #
            if summonerId not in summonerIdToData:
                summonerIdToData[summonerId] = defaultdict(set)

            champId = matchIdToData[matchId]["participants"][summonerId]["championId"]
            summonerIdToData[summonerId][champId].add(matchId)

        # mastery
        mastery = getChampionMastery(summonerId, championId)

        # winRate
        matchIdList = sorted(
            list(summonerIdToData[summonerId][championId]),
            key=lambda x: matchIdToData[x]["gameCreation"],
            reverse=True,
        )
        start = matchIdList.index(targetMatchId)
        winRate = getWinRate(matchIdList[start + 1 :], summonerId)

        temp_dict[teamAndLine] = {
            "championId": championId,
            "winRate": winRate,
            "mastery": mastery,
            "numOfPlay": len(matchIdList[start + 1 :]),
        }

        targetSummonerQueue.append(summonerId)

    # done
    if done:
        data_dict[targetMatchId] = temp_dict.copy()
        summoner_done.add(targetSummonerId)
        matchId_done.add(targetMatchId)


file_name = f"MasteryAndWinRate_{targetTier}_{targetDivision}_{len(data_dict)}.json"

with open(os.path.join(data_path, file_name), "w") as fp:
    json.dump(data_dict, fp)

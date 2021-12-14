from riotwatcher import LolWatcher, ApiError
from pprint import pprint
from tqdm import tqdm
import datetime
from contextlib import contextmanager

import pandas as pd
import sys
import os
import csv
import json
import time

from collections import defaultdict

KST = datetime.timezone(datetime.timedelta(hours=9))
data_path = "./data"

"""
for fileName in os.listdir(data_path):
    if "Summoner_Info" in fileName:
        summoner_df = pd.read_csv(os.path.join(data_path, fileName))
        break
"""


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    # print(f"[{name}] done in {time.time() - t0:.3f} s") # 출력 안 되게 수정


my_region = "kr"
queue = "RANKED_SOLO_5x5"
queueId = 420  # "5v5 Ranked Solo games" ID -> https://static.developer.riotgames.com/docs/lol/queues.json

# tiers = ['DIAMOND', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'IRON']
# top_tier = ["CHALLENGER", "GRANDMASTER", "MASTER"]
# divisions = ['I', 'II', 'III', 'IV']

targetTier = "IRON"  # 티어 넣어주세요
targetDivision = "IV"  # 구간 넣어주세요


def collectPuuid(summoner_df, targetTier, targetDivision, riot_token):
    lol_watcher = LolWatcher(riot_token)
    print(f"Collect Puuid")
    print(f"    Target tier : {targetTier} {targetDivision}")
    summonerNameList = summoner_df[
        (summoner_df.tier == targetTier) & (summoner_df.division == targetDivision)
    ]["summonerName"]
    puuids = []

    with timer(f"{targetTier} {targetDivision} : puuid"):
        for summonerName in tqdm(summonerNameList, file=sys.stdout):
            try:
                puuid = lol_watcher.summoner.by_name(my_region, summonerName)["puuid"]
            except:
                continue
            puuids.append(puuid)

    print()
    return puuids


def collectMatchData(
    matchDataDict,
    puuidList,
    targetTier,
    targetDivision,
    continueArgs=None,
    riot_token=None,
):
    lol_watcher = LolWatcher(riot_token)
    if continueArgs is None:
        assert len(matchDataDict) == 0, "수집 완료된 Dict 입니다."
    print(f"Collect Match data")
    print(f"    Target tier : {targetTier} {targetDivision}")

    now = datetime.datetime.now(KST)

    with timer(f"{targetTier} {targetDivision} : Match Data"):

        if continueArgs is not None:
            puuids = puuidList[continueArgs[0] :]
            matchIdSet = continueArgs[1]
        else:
            puuids = puuidList[:]
            matchIdSet = set()  # 중복 수집 방지

        data_cnt = sum(len(data) for data in matchDataDict.values())
        pbar = tqdm(puuids, file=sys.stdout, total=len(puuids))
        for puuid in pbar:

            if continueArgs is not None:
                startIndex = continueArgs[2]
            else:
                startIndex = 0

            count = 100

            while True:
                try:
                    matchIdList = lol_watcher.match.matchlist_by_puuid(
                        "asia", puuid, start=startIndex, count=count, queue=queueId
                    )  # 매치 아이디
                except ApiError as err:
                    return (
                        matchDataDict,
                        now,
                        (puuidList.index(puuid), matchIdSet, startIndex, err.response),
                    )

                if len(matchIdList) == 0:
                    break

                for i, matchId in enumerate(matchIdList):
                    if matchId in matchIdSet:
                        continue  # 중복 수집 방지
                    matchIdSet.add(matchId)

                    try:
                        matchData = lol_watcher.match.by_id("asia", matchId)  # 매치 데이터
                    except ApiError as err:
                        return (
                            matchDataDict,
                            now,
                            (
                                puuidList.index(puuid),
                                matchIdSet,
                                startIndex + i,
                                err.response,
                            ),
                        )

                    version = matchData["info"]["gameVersion"]
                    matchDataDict[version].append(matchData)

                    data_cnt += 1
                    pbar.set_description("    " + f"current MatchData -> {data_cnt}개")

                startIndex += count

            # return matchDataDict, now, None # 한명만 수집할 때

    print()
    return matchDataDict, now, None


"""
riot_token = input("Enter riot API key to start: ")

# puuidList 수집
puuidList = collectPuuid(summoner_df, targetTier, targetDivision, riot_token)


# puuidList csv로 저장하기
puuids = ["puuids"]
with open(
    os.path.join(data_path, f"{targetTier}_{targetDivision}_puuids.csv"), "w"
) as f:
    writer = csv.writer(f, delimiter="\n")
    writer.writerow(puuids)
    writer.writerow(puuidList)
"""

# csv 로드해서 puuidList 만들기
puuidList_data = []
with open(
    os.path.join(data_path, f"{targetTier}_{targetDivision}_puuids.csv"), "r"
) as f:
    reader = csv.reader(f)
    for row in reader:
        puuidList_data.append(row[0])
puuidList = puuidList_data[1:]


# 해당 티어 특정 순위권 유저만 수집
puuidList = puuidList[:10]


# Match Data 수집 : error 403 -> API key 재발급 후 입력
#                  error 503 -> 자동으로 재수집
matchDataDict = defaultdict(list)
continueArgs = None
riot_token = input("Enter riot API key to start: ")

while True:
    matchDataDict, now, continueArgs = collectMatchData(
        matchDataDict, puuidList, targetTier, targetDivision, continueArgs, riot_token
    )

    if continueArgs is not None:
        if continueArgs[-1].json()["status"]["status_code"] == 403:
            riot_token = input(
                "\033[31m" + "403 Error, Enter riot API key to continue: " + "\033[0m"
            )
            continue
        if continueArgs[-1].json()["status"]["status_code"] == 503:
            print("\033[32m" + "503 Error, restarting..." + "\033[0m")
            continue
    else:
        break

file_name = f"{targetTier}_{targetDivision}_MatchData_{now.year}_{now.month}_{now.day}_{now.hour}h_{now.minute}m_{now.second}s.json"

with open(os.path.join(data_path, file_name), "w") as fp:
    json.dump(matchDataDict, fp)
    print("Match data has been collected!")

import time

import requests
from riotwatcher import LolWatcher, ApiError
import pandas as pd
BASE_URL = 'https://kr.api.riotgames.com'
RESION = 'ko_KR'
TOKEN = 'RGAPI-6a7b707e-6a76-478c-a8a6-183dc469b10f'

headers = {
    #"Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,pt;q=0.6",
    #"Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    #"Origin": "https://developer.riotgames.com",
    "X-Riot-Token": TOKEN
}

def getSpectatorInfo(encryptedSummonerId):
    url = BASE_URL + f'/lol/spectator/v4/active-games/by-summoner/{encryptedSummonerId}'
    return requests.get(url, headers=headers)

def getSpectatorInfo2(summonerName):
    url = BASE_URL + f'/lol/summoner/v4/summoners/by-name/{summonerName}'
    encryptedSummonerId = requests.get(url, headers=headers).json()
    if 'id' not in encryptedSummonerId:
        return False
    # print(encryptedSummonerId)
    # proceed = 0
    # while('id' not in encryptedSummonerId):
    #     time.sleep(0.5)
    #     encryptedSummonerId = requests.get(url, headers=headers).json()
    #     proceed +=1
    #     if proceed==9:
    #         return False
    encryptedSummonerId = encryptedSummonerId['id']
    url = BASE_URL + f'/lol/spectator/v4/active-games/by-summoner/{encryptedSummonerId}'
    return requests.get(url, headers=headers).json()


def get_champion_info():
    lol_watcher = LolWatcher(TOKEN)
    version = get_ddragon_recent_version()
    static_champ_list = lol_watcher.data_dragon.champions(version, True, RESION )
    champ_dict = {}
    for key in static_champ_list['data']:
        row = static_champ_list['data'][key]
        champ_dict[row['key']] = row['id']

    # print dataframe
    df = pd.DataFrame(static_champ_list['data'])
    df.columns = [i for i in range(len(df.keys()))]
    df = df.transpose()
    df = df.loc[:, ["key", "name", "image"]]
    df["image"] = df["image"].apply(lambda x: x['full'])

    return df

def get_ddragon_recent_version():
    return requests.get(url="https://ddragon.leagueoflegends.com/api/versions.json").json()[0]

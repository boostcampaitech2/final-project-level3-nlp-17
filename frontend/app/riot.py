import time

import requests

BASE_URL = 'https://kr.api.riotgames.com'
RESION = 'ko_KR'
TOKEN = 'RGAPI-874a6f12-cbfc-4e0f-86a2-16d751bbb7c8'

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
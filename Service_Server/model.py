import requests

from riot import getSpectatorInfo2

inference_server_ip = ''

def model_inference(summonerName):
    
    reponse = getSpectatorInfo2(summonerName)
    if reponse.status_code != 200: return reponse

    spectatorInfo = reponse.json()
    
    data = {
        'matchId' : 'KR_' + spectatorInfo['gameId']
    }

    lines = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

    team = {100:[], 200:[]}
    for participant in spectatorInfo['participants']:
        team[participant['teamId']].append(participant)

    for teamId in team.keys():
        for participant, line in zip(team[teamId], lines):
            key = '_'.join([teamId, line])
            data[key] = (participant['summonerId'], participant['championId'])


    # infernce
    url = inference_server_ip + '/model/inference/by-matchInfo'
    reponse = requests.post(url=url,data=data)

    if reponse.status_code != 200: return reponse

    return reponse.json()
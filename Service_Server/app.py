from fastapi import FastAPI
from model import model_inference

from riot import getSpectatorInfo, getSpectatorInfo2

app = FastAPI()

@app.get('/')
def root():
    return {'hello':'world'}

@app.get('/riot/spectator/by-summonerId/{encryptedSummonerId}')
def getSpectator(encryptedSummonerId):
    return getSpectatorInfo(encryptedSummonerId).json()

@app.get('/riot/spectator/by-name/{summonerName}')
def getSpectatorByName(summonerName):
    return getSpectatorInfo2(summonerName).json()

@app.get('/model/by-name/{summonerName}')
def getInference(summonerName):
    return model_inference(summonerName)

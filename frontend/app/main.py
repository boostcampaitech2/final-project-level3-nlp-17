import json

from starlette.responses import RedirectResponse
from fastapi import FastAPI, Form, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from .riot import getSpectatorInfo2, get_champion_info
from pprint import pprint
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Optional
import requests
# init settings
app = FastAPI()

app.mount("/statics", StaticFiles(directory="./statics"), name="static")
templates = Jinja2Templates(directory="./templates")

import inspect
from typing import Type

from fastapi import Form
from pydantic import BaseModel
from pydantic.fields import ModelField

@app.get("/", response_class=HTMLResponse)
async def get_index_page(request: Request):
    print("Get")
    predicts = -1
    return templates.TemplateResponse('index.html', context={'request': request, 'predict': predicts})


@app.get("//riot.txt", response_class=HTMLResponse)
async def get_riot(request: Request):
    print("Get")
    return ""

@app.post("/search", response_class=HTMLResponse)
async def get_user_info(request: Request,
                        _200_TOP: str = Form(...),
                        _200_JUNGLE: str = Form(...),
                        _200_MIDDLE: str = Form(...),
                        _200_BOTTOM: str = Form(...),
                        _200_UTILITY: str = Form(...),
                        _100_TOP: str = Form(...),
                        _100_JUNGLE: str = Form(...),
                        _100_MIDDLE: str = Form(...),
                        _100_BOTTOM: str = Form(...),
                        _100_UTILITY: str = Form(...),
                        _200_TOP_: str = Form(...),
                        _200_JUNGLE_: str = Form(...),
                        _200_MIDDLE_: str = Form(...),
                        _200_BOTTOM_: str = Form(...),
                        _200_UTILITY_: str = Form(...),
                        _100_TOP_: str = Form(...),
                        _100_JUNGLE_: str = Form(...),
                        _100_MIDDLE_: str = Form(...),
                        _100_BOTTOM_: str = Form(...),
                        _100_UTILITY_: str = Form(...),
                        userid: str = Form(...)):

    data = {}
    data["100_TOP"] = [_100_TOP, _100_TOP_]
    data["100_JUNGLE"] = [_100_JUNGLE, _100_JUNGLE_]
    data["100_MIDDLE"] = [_100_MIDDLE, _100_MIDDLE_]
    data["100_BOTTOM"] = [_100_BOTTOM, _100_BOTTOM_]
    data["100_UTILITY"] = [_100_UTILITY, _100_UTILITY_]

    data["200_TOP"] = [_200_TOP, _200_TOP_]
    data["200_JUNGLE"] = [_200_JUNGLE, _200_JUNGLE_]
    data["200_MIDDLE"] = [_200_MIDDLE, _200_MIDDLE_]
    data["200_BOTTOM"] = [_200_BOTTOM, _200_BOTTOM_]
    data["200_UTILITY"] = [_200_UTILITY, _200_UTILITY_]
    pprint(data)
    userinfo = getSpectatorInfo2(userid)
    red_team, blue_team, result, user_color, userinfo = formatting_to_inference(userinfo, userid, data)
    print("*"*50)
    pprint(result)
    result = json.dumps({"summonerDict": data, "matchId": userinfo})
    pprint(red_team)
    predict = requests.post("http://101.101.216.156:6013/model/inference/by-matchInfo", data=result).json()
    #predict = 0.3
    print(predict)
    return templates.TemplateResponse('index.html', context={
        'request': request, 'predict': predict['output'][0][user_color],
        'red_team': red_team, 'blue_team': blue_team, "user_color": user_color, "userid": userid, "matchID": userinfo})


@app.post("/result", response_class=HTMLResponse)
async def post_result(request: Request, userid: str = Form(...)):
    userinfo = getSpectatorInfo2(userid)
    pprint(userinfo)
    if userinfo == False:
        print("user find fail.")
        return RedirectResponse(url="/", status_code=302)
    if 'gameId' not in userinfo:
        print("user find fail.")
        return RedirectResponse(url="/", status_code=302)
    pprint("user Find ok")
    red_team, blue_team, data, user_color, userinfo = formatting_to_inference(userinfo, userid)
    data = json.dumps({"summonerDict": data, "matchId": userinfo})
    pprint(data)
    predict = requests.post("http://101.101.216.156:6013/model/inference/by-matchInfo", data=data).json()
    #predict = 0.3
    print(predict)
    return templates.TemplateResponse('index.html', context={
        'request': request, 'predict': predict['output'][0][user_color],
        'red_team': red_team, 'blue_team': blue_team, "user_color": user_color, "userid":userid , "matchID": userinfo})


@app.exception_handler(StarletteHTTPException)
async def my_custom_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return templates.TemplateResponse('404.html', {'request': request})
    # else:
    #     # Just use FastAPI's built-in handler for other errors
    #     return await http_exception_handler(request, exc)

def formatting_to_inference(userinfo, userid, change: Optional[int] = None):
    df = get_champion_info()
    data = {}
    red_team = {}
    blue_team = {}
    lines = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    team = {100: [], 200: []}
    user_color = 0
    # pprint(userinfo)
    for participant in userinfo['participants']:
        team[participant['teamId']].append(participant)
        if participant['summonerName'] == userid:
            user_color = int((participant['teamId'] - 100) / 100)
    for teamId in team.keys():
        for participant, line in zip(team[teamId], lines):
            key = "_".join([str(teamId), str(line)])
            if change is None:
                data[key] = [participant["summonerId"].replace("'", '"'), participant["championId"]]
            else:
                data[key] = change[key]
            champ = df.loc[df["key"] == str(data[key][1])].iloc[0]

            if teamId == 100:  # 파랑색
                blue_team[key] = [champ['name'], champ['image'], str(line), participant['summonerName'],
                                  data[key][0], data[key][1]]
            else:  # 붉은색
                red_team[key] = [champ['name'], champ['image'], str(line), participant['summonerName'],
                                 data[key][0], data[key][1]]


    return red_team, blue_team, data, user_color, "KR_" + str(userinfo['gameId'])
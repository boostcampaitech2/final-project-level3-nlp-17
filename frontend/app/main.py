import json

from starlette.responses import RedirectResponse
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from .riot import getSpectatorInfo, getSpectatorInfo2, get_champion_info
import requests
from pprint import pprint
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exception_handlers import http_exception_handler

# init settings
app = FastAPI()

app.mount("/statics", StaticFiles(directory="./statics"), name="static")
templates = Jinja2Templates(directory="./templates")

@app.get("/", response_class=HTMLResponse)
async def get_index_page(request: Request):
    print("Get")
    predicts = -1
    return templates.TemplateResponse('index.html', context={'request': request, 'predict': predicts})


@app.post("/result", response_class=HTMLResponse)
async def post_result(request: Request, userid: str = Form(...)):
    userinfo = getSpectatorInfo2(userid)
    if userinfo == False:
        return RedirectResponse(url="/", status_code=302)
    if 'gameId' not in userinfo:
        return RedirectResponse(url="/", status_code=302)
    pprint("user Find ok")
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

    print(user_color)
    for teamId in team.keys():
        for participant, line in zip(team[teamId], lines):
            key = "_".join([str(teamId), str(line)])
            data[key] = [participant["summonerId"].replace("'", '"'), participant["championId"]]
            champ = df.loc[df["key"] == str(participant["championId"])].iloc[0]

            if teamId == 100: # 파랑색
                blue_team[key] = [champ['name'], champ['image'], str(line), participant['summonerName']]
            else: # 붉은색
                red_team[key] = [champ['name'], champ['image'], str(line), participant['summonerName']]

    data = json.dumps({"summonerDict": data, "matchId": "KR_" + str(userinfo['gameId'])})
    predict = requests.post("http://101.101.216.156:6013/model/inference/by-matchInfo", data=data).json()
    # predict = 0.7
    print(predict)
    return templates.TemplateResponse('index.html', context={
        'request': request, 'predict': predict['output'][0][user_color],
        'red_team': red_team, 'blue_team': blue_team, "user_color": user_color})


@app.exception_handler(StarletteHTTPException)
async def my_custom_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return templates.TemplateResponse('404.html', {'request': request})
    # else:
    #     # Just use FastAPI's built-in handler for other errors
    #     return await http_exception_handler(request, exc)

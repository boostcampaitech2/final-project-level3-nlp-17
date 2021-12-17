from starlette.responses import RedirectResponse
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from .riot import getSpectatorInfo, getSpectatorInfo2
import requests

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
    #print(userinfo)
    if userinfo == False:
        return RedirectResponse(url="/", status_code=302)
    if 'gameId' not in userinfo:
        return RedirectResponse(url="/", status_code=302)
    print("user Find ok")
    #print(userinfo)
    #predicts = requests.post("http://localhost:8001/order", data=userinfo)
    predict = 0.7
    return templates.TemplateResponse('index.html', context={'request': request, 'predict':predict})

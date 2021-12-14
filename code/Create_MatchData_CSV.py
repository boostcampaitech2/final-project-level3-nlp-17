from contextlib import contextmanager

import json
import os

from collections import defaultdict, deque
from pprint import pprint
from tqdm import tqdm
import time

import pandas as pd
import numpy as np
import argparse

match_columns = {
    "data" : set(['assists', 'baronKills', 'damageDealtToBuildings', 'damageDealtToObjectives', 'damageDealtToTurrets', 'damageSelfMitigated',
                    'deaths', 'dragonKills', 'goldEarned', 'goldSpent', 'kills', 'magicDamageDealt', 'magicDamageDealtToChampions', 'magicDamageTaken',
                    'neutralMinionsKilled', 'physicalDamageDealt', 'physicalDamageDealtToChampions', 'physicalDamageTaken', 'totalDamageDealt',
                    'totalDamageDealtToChampions', 'totalDamageShieldedOnTeammates', 'totalDamageTaken', 'totalHeal', 'totalMinionsKilled',
                    'totalTimeCCDealt', 'totalTimeSpentDead', 'trueDamageDealt', 'trueDamageDealtToChampions', 'trueDamageTaken', 'turretKills', 'turretsLost',
                    'visionScore', 'visionWardsBoughtInGame', 'wardsKilled', 'wardsPlaced']),
    "info" : set(['championName', 'championId', 'teamPosition', 'win']),
}

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')


@contextmanager
def timer(process):
    t0 = time.time()
    print(process, '...')
    yield
    print(f"[{process}] done in {time.time() - t0:.3f} s\n")

def preprocessingRawData(rawMatchData, args):
    
    if args.change_game_duration:
        targetVersion = [version for version in rawMatchData.keys() if int(version[3:]) < 20]
        for version in targetVersion:
            for i in range(len(rawMatchData[version])):
                rawMatchData[version][i]['info']['gameDuration'] /= 1000
    
    matchIdToData = {}

    for matchData in [data for dataList in rawMatchData.values() for data in dataList]:
        matchId = matchData['metadata']['matchId']
        assert matchId not in matchIdToData, "중복된 매치 데이터가 있습니다."
        matchIdToData[matchId] = matchData
        
    summonerIdToDict = {}

    for matchId, matchData in matchIdToData.items():
        for participant in matchData['info']['participants']:
            summonerId = participant['summonerId']
            if summonerId not in summonerIdToDict:
                summonerIdToDict[summonerId] = defaultdict(list)
                
            key = '_'.join([participant['teamPosition'], str(participant['championId'])])
            
            participant['matchId'] = matchId
            participant['gameCreation'] = matchData['info']['gameCreation']
            participant['gameDuration'] = matchData['info']['gameDuration']
            summonerIdToDict[summonerId][key].append(participant)
    
    
    return matchIdToData, summonerIdToDict

# typeOfMatch 
# s : 특정 소환사 매치만
# a : 모든 매치 정보
# l : label Data
def createDict(info=True, numOfSummoner=None, typeOfMatch=['s'], init=None):
    columns = []
    
    if info:                 info_columns = match_columns['info']
    elif 'l' in typeOfMatch: info_columns = ['win']
    else:                    info_columns = []
    
    if numOfSummoner is not None:
        for num, mType in [(num, mType) for num in range(numOfSummoner) for mType in typeOfMatch]:
            for column in [column for columns in [match_columns['data'], info_columns] for column in columns]:
                columns.append('_'.join([column, str(num), mType]))
    else:
        columns.extend([column for columns in [match_columns['data'], info_columns] for column in columns])
    
    if init is not None: return {column:init for column in columns}
    else:                return {column:[] for column in columns}

def divAndAddMatchData(dataDict, matchDataList, i, typeOfMatch, referGameNum):
    sumDict = createDict(info=True, init=0)

    start = 1               #
    end = start + referGameNum # 20경기 까지
    
    # data
    for matchData in matchDataList[start:end]:    
        gameDuration = matchData['gameDuration'] / 60 # 분 단위
        for key, value in matchData.items():
            if key not in match_columns['data']: continue
            if key in sumDict: sumDict[key] += value / gameDuration / len(matchDataList[start:end]) # 평균   
            
    # info
    for key, value in matchDataList[0].items():
        if key not in match_columns['info']: continue
        if key in sumDict: sumDict[key] = value
        
    # 합산 데이터 추가
    for key, value in sumDict.items():
        key = '_'.join([key, str(i), typeOfMatch])
        if key in dataDict: dataDict[key].append(value)
    
    return dataDict

def makeDataset(matchIdToData, summonerIdToDict, args):
    line = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    teamAndLines = ['100_TOP', '100_JUNGLE', '100_MIDDLE', '100_BOTTOM', '100_UTILITY', 
                '200_TOP', '200_JUNGLE', '200_MIDDLE', '200_BOTTOM', '200_UTILITY']
    
        
    dataDict = createDict(info=args.include_info, numOfSummoner=10, typeOfMatch=['s'])
    dataDict['matchId'] = []

    labelDict = createDict(info=args.include_info, numOfSummoner=10, typeOfMatch=['l'])
    labelDict['matchId'] = []

    timeLog = []
    pbar = tqdm(matchIdToData.items())
    for matchId, matchData in pbar:
        
        teamIdToData = defaultdict(dict)
        for participant in matchData['info']['participants']:
            teamIdToData['_'.join([str(participant['teamId']), participant['teamPosition']])] = participant 
            
        assert len(teamIdToData) == 10
        
        # 매치 데이터 체크
        matchCheck = True
        for participant in teamIdToData.values():
            if participant['summonerId'] not in summonerIdToDict: matchCheck = False
            if participant['teamPosition'] not in line:           matchCheck = False
        if not matchCheck: continue
        
        t0 = time.time()
        for i, team in enumerate(teamAndLines):
            participant = teamIdToData[team]
            
            summonerId = participant['summonerId']
            lineAndChamp = '_'.join([participant['teamPosition'], str(participant['championId'])])
            
            # 특정 소환사 / 특정 라인 / 특정 챔피언 
            matchDataList = sorted(summonerIdToDict[summonerId][lineAndChamp], key=lambda x: x['gameCreation'], reverse=True)
            start = [matchData['matchId'] for matchData in matchDataList].index(matchId)
            dataDict = divAndAddMatchData(dataDict, matchDataList[start:], i, 's', args.refer_num)
            
            # label data
            gameDuration = participant['gameDuration'] / 60
            for key, value in participant.items():
                if key in match_columns['data']: value /= gameDuration
                key = '_'.join([key, str(i), 'l'])
                if key in labelDict: labelDict[key].append(value)
                
        dataDict['matchId'].append(matchId) 
        labelDict['matchId'].append(matchId)
        
        createTime = time.time() - t0
        timeLog.append(createTime)
        pbar.set_description(f'Curr : {createTime:.6f}s, Mean : {sum(timeLog) / len(timeLog):.6f}s, Max : {max(timeLog):.6f}s, Min : {min(timeLog):.6f} ')
    
    return (dataDict, labelDict)
        
        
def postprocessingDataset(dataset):
    data_df = pd.DataFrame(dataset[0])
    label_df = pd.DataFrame(dataset[1])
    
    # info column _(언더바) 추가
    for df_column in label_df.columns.tolist():
        if df_column.split('_')[0] in match_columns['info']:
            label_df.rename(columns={df_column:'_'+df_column}, inplace = True)
    
    data_df.rename(columns={'matchId':'_matchId'}, inplace = True)
    label_df.rename(columns={'matchId':'_matchId'}, inplace = True)
    
    # win column 정수형으로 변경
    for df_column in label_df.columns.tolist():
        if 'win' in df_column:
            label_df[df_column] = np.where(label_df[df_column] == True, 1, 0)
            
    return (data_df, label_df)
            

def main(args):
    # Load raw data
    with timer('Raw data loading'):
        file_name = os.path.join(args.data_path, args.raw_data + '.json')
        with open(file_name, 'r') as fp:
            rawMatchData = json.load(fp)
    
    with timer('Preprocessing raw data'):
        matchIdToData, summonerIdToDict = preprocessingRawData(rawMatchData, args)
        
    print(f'Num of matchData : {len(matchIdToData)}\n')
    
    with timer('Make Dataset'):
        dataset = makeDataset(matchIdToData, summonerIdToDict, args)
    
    with timer('Postprocessing dataset'):
        dataset_df = postprocessingDataset(dataset)
    
    # save
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    with timer('Save'):
        save_path = os.path.join(args.save_path, "DATA_" + args.save_name + '.csv')
        dataset_df[0].to_csv(save_path, index=False)
        
        save_path = os.path.join(args.save_path, "LABEL_" + args.save_name + '.csv')
        dataset_df[1].to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make csv dataset")
    
    parser.add_argument('--data_path', default='./data', type=str, help='Data directory path')
    parser.add_argument('--raw_data', 
                       default='CHALLENGER_I_MatchData_2021_11_28_0h_29m_35s',
                       type=str, help='Raw data name')
    parser.add_argument('--refer_num', default=20, type=int, help='Num of game to refer')
    parser.add_argument('--save_path', default='./dataset', type=str, help='Save path')
    parser.add_argument('--save_name', default='', type=str, help='Save file name')
    parser.add_argument('--include_info', default=True, type=str2bool, help='Include info columns')
    parser.add_argument('--change_game_duration', default=True, type=str2bool, help='Preprocessing gameDuration')
    
    args = parser.parse_args()
    
    print(f'include_info : {args.include_info}')
    
    if args.save_name == '': args.save_name = args.raw_data
    
    main(args)
    
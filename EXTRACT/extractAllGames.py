import GetAllGamesCommon
import GetAllLeagueCommon
import GetAllLeagueBothPlata
from importlib import reload
import os
from dotenv import load_dotenv

load_dotenv()

def extractStats(division, season, targetTeam, jorFirst, jorLast, sDir, chromeDriver):
    print(division)
    divSplit = division.split(',')[0]
    try:
        groupSplit = division.split(',')[1]
    except:
        pass

    if division == 'ENDESA' or division == 'LF':
        html_name = 'lfendesa'
        division = 'DIA'
    if division == 'ORO' or division.split(',')[0] == 'ORO':
        html_name = 'ligaleboro'
        groupFeb = '1'
    elif division == 'DIA':
        html_name = 'lfendesa'
        groupFeb = '4'
    elif divSplit == 'PLATA':
        html_name = 'ligalebplata'
        bUnaFase = False
        if len(division.split(',')) == 3:
            if int(season) > 2017:
                if division.split(',')[2] == 'A1':
                    groupFeb = '2'
                else:
                    groupFeb = '18'
            else:
                groupFeb = '2'
        else:
            bUnaFase = True
            if int(season) > 2017:
                if division.split(',')[1] == 'ESTE':
                    groupFeb = '2'
                else:
                    groupFeb = '18'
            else:
                groupFeb = '2'
    elif divSplit == 'EBA':
        html_name = 'ligaeba'
        if groupSplit[0] == 'A': # AA AB AC
            if int(season) > 2019:
                if groupSplit[1] == 'A':
                    groupFeb = '3'
                else:
                    groupFeb = '17'
            else:
                groupFeb = '3'
        elif groupSplit[0] == 'B': # BA BBA
            if int(season) > 2019:
                if int(season) < 2021:
                    if groupSplit[1] == 'A':
                        groupFeb = '5'
                    else:
                        groupFeb = '57'
                else:
                    groupFeb = '5'
            else:
                groupFeb = '5'
        elif groupSplit[0] == 'C': # C1 C2 C3
            if int(season) > 2019:
                if groupSplit[1] == '1':
                    groupFeb = '48'
                elif groupSplit[1] == '2':
                    groupFeb = '49'
                elif groupSplit[1] == '3':
                    groupFeb = '59'
                elif groupSplit[1] == '4':
                    groupFeb = '60'
                elif groupSplit[1] == '5':
                    groupFeb = '61'
            elif int(season) > 2018:
                if groupSplit[1] == 'A':
                    groupFeb = '6'
                elif groupSplit[1] == 'B' or groupSplit[1] == '2':
                    groupFeb = '46'
                elif groupSplit[1] == 'C' or groupSplit[1] == '3':
                    groupFeb = '46'
            else:
                groupFeb = '6'
        elif groupSplit[0] == 'D': # DA DB
            if int(season) > 2019:
                if groupSplit[1] == 'A':
                    groupFeb = '7'
                else:
                    groupFeb = '47'
            else:
                groupFeb = '7'
        elif groupSplit[0] == 'E': # EA EB
            if int(season) > 2019:
                if groupSplit[1] == 'A':
                    groupFeb = '8'
                elif groupSplit[1] == 'B':
                    groupFeb = '39'
                elif groupSplit[1] == 'C':
                    groupFeb = '53'
            else:
                groupFeb = '8'
    elif divSplit == 'LF2': # A B C
        html_name = 'ligaeba'
        if int(season) > 2019:
            if groupSplit == 'A':
                groupFeb = '9'
            elif groupSplit == 'B':
                groupFeb = '10'
            elif groupSplit == 'C':
                groupFeb = '56'
        else:
            groupFeb = '9'
    elif divSplit == 'LFCHALLENGE':  # A B C
        html_name = 'lfchallenge'
        groupFeb = '67'

    html_doc = "https://baloncestoenvivo.feb.es/calendario/" + html_name + '/' + groupFeb + '/' + season
    print(html_doc)
    if division == 'ORO' or division == 'DIA' or division == 'ENDESA' or division == 'LF' or division == 'LFCHALLENGE':
        GetAllLeagueCommon.extractStatisticsAllLeague(html_doc, 'Liga'+division.replace(',','-'), season, jorFirst, jorLast, division, sDir, chromeDriver, 1, '', False, division, '', '', 'Castellano', False)
    elif divSplit == 'ORO':
        GetAllLeagueCommon.extractStatisticsAllLeague(html_doc, 'Liga'+division.replace(',','-'), season, jorFirst, jorLast, division.split(',')[1], sDir, chromeDriver, 1, '', False, division, '', '', 'Castellano', False)
    elif divSplit == 'PLATA':
        if bUnaFase == False:
            GetAllLeagueBothPlata.extractStatisticsPlataAll(html_doc,targetTeam,season,jorFirst,jorLast,division.split(',')[1],division.split(',')[2],sDir,chromeDriver,1,'',False,division,'','', 'Castellano', False)
            reload(GetAllLeagueBothPlata)
        else:
            GetAllLeagueCommon.extractStatisticsAllLeague(html_doc, 'Liga'+division.replace(',','-'), season, jorFirst, jorLast, division.split(',')[1], sDir, chromeDriver, 1, '', False, '', 'Fase1', '', 'Castellano', False)
    elif divSplit == 'EBA':
        GetAllLeagueCommon.extractStatisticsAllLeague(html_doc, 'Liga'+division.replace(',','-'), season, jorFirst, jorLast, division.split(',')[1], sDir, chromeDriver, 1, '', False, division, '', '', 'Castellano', False)
    elif divSplit == 'LF2':
        GetAllLeagueCommon.extractStatisticsAllLeague(html_doc, 'Liga'+division.replace(',','-'), season, jorFirst, jorLast, division.split(',')[1], sDir, chromeDriver, 1, '', False, division, '', '', 'Castellano', False)
    reload(GetAllLeagueCommon)

path = os.getenv("CSV_OUT")

jornadas = {
    2024 : 30,
    2023 : 30,
    2022 : 30,
    2021 : 30,
    2020 : 30,
    2019 : 26,
    2018 : 26,
    2017 : 26,
    2016 : 26,
    2015 : 26,
    2014 : 26,
    2013 : 22,
    2012 : 22,
    2011 : 26,
    2010 : 26,
    2009 : 26,
    2008 : 26,
    2007 : 26,
    2006 : 26,
    2005 : 26,
    2004 : 26,
    2003 : 26,
    2002 : 26,
    2001 : 26,
    2000 : 26,
    1999 : 26,
    1998 : 26,
    1997 : 22,
}

for year, jornada in jornadas.items():
    print(f'Extrayendo: {year}/{year+1}')
    os.mkdir(path + f'Reports/LF/{year}')
    extractStats('ENDESA', str(year), 'Liga', 1,jornada, path + f'Reports/LF/{year}', path + 'chromedriver')
    print('----------------------------------')
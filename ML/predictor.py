from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
import pandas as pd
import time
from model import Model

class Predictor:
    driver = None
    options = None
    service = None
    teamIDs = {'ATL': '1610612737', 'BOS': '1610612738', 'BKN': '1610612751', 'CHA': '1610612766',
        'CHI': '1610612741', 'CLE': '1610612739', 'DAL': '1610612742', 'DEN': '1610612743',
        'DET': '1610612765', 'GSW': '1610612744', 'HOU': '1610612745', 'IND': '1610612754',
        'LAC': '1610612746', 'LAL': '1610612747', 'MEM': '1610612763', 'MIA': '1610612748',
        'MIL': '1610612749', 'MIN': '1610612750', 'NOP': '1610612740', 'NYK': '1610612752',
        'OKC': '1610612760', 'ORL': '1610612753', 'PHI': '1610612755', 'PHX': '1610612756',
        'POR': '1610612757','SAC': '1610612758', 'SAS': '1610612759', 'TOR': '1610612761',
        'UTA': '1610612762', 'WAS': '1610612764'}
    columns = ['HOME PTS', 
               'HOME FGM', 'HOME FGA', 'HOME FG%', 'HOME 3PM', 'HOME 3PA', 'HOME 3P%', 
               'HOME FTM', 'HOME FTA', 'HOME FT%', 'HOME OREB', 'HOME DREB', 'HOME REB', 
               'HOME AST', 'HOME TOV', 'HOME STL' ,'HOME BLK', 'HOME PF', 'HOME +/-',
               'AWAY PTS', 'AWAY FGM', 'AWAY FGA', 'AWAY FG%', 'AWAY 3PM', 'AWAY 3PA', 
               'AWAY 3P%', 'AWAY FTM', 'AWAY FTA', 'AWAY FT%', 'AWAY OREB', 'AWAY DREB', 
               'AWAY REB', 'AWAY AST', 'AWAY TOV', 'AWAY STL' ,'AWAY BLK', 'AWAY PF', 
               'AWAY +/-']

    def __init__(self):
        self.options = webdriver.ChromeOptions()
        # self.options.add_argument("--headless=new") # Run without UI popup (DOESNT WORK)
        self.options.add_argument("--ignore-certificate-errors")
        self.options.add_argument("--log-level=3")  # Suppresses most logs
        self.options.add_argument("--disable-logging")

    def getUserInput(self):
        homeTeam = input("Enter Home Team: ")
        awayTeam = input("Enter Away Team: ")
        modelSel = input("Specify Model (svm, knn, logistic): ")
        return homeTeam, awayTeam, modelSel

    def goToPage(self, teamID):
        try:
            url = 'https://www.nba.com/stats/team/' + teamID + '/traditional'
            self.driver.get(url)
        except Exception as e:
            print(f'Exception occured while changing pages \n{e}')

    def getStats(self):
        columns = []
        table = []
        tableHead_xpath = '//div[3]/table/thead//th'
        tableRow_xpath = '//div[3]/table/tbody/tr/td'

        try:
            NUM_COLUMNS = len(self.driver.find_elements(By.XPATH, tableHead_xpath))
            tableRow = self.driver.find_elements(By.XPATH, tableRow_xpath)
            rowStats = []

            # Loop through columns
            for i in range(3, NUM_COLUMNS):
                statName_xpath = tableHead_xpath + '[' + str(i + 1) + ']'
                statName = self.driver.find_element(By.XPATH, statName_xpath)
                columns.append(statName.text)

                stat_xpath = tableRow_xpath + '[' + str(i + 1) + ']'
                stat = self.driver.find_element(By.XPATH, stat_xpath)
                rowStats.append(stat.text)

            table.append(rowStats)
            df = pd.DataFrame(table, columns=columns)
        except Exception as e:
            print(f'Error while getting team stats \n{e}')

        return df
    
    def scrapeTeams(self, homeTeam, awayTeam):

        # Web Scrape Home Team Stats
        self.driver = webdriver.Chrome(options=self.options)
        self.goToPage(self.teamIDs[homeTeam])
        time.sleep(10)
        dfHome = self.getStats()
        self.driver.close()

        # Web Scrape Away Team Stats
        self.driver = webdriver.Chrome(options=self.options)
        self.goToPage(self.teamIDs[awayTeam])
        time.sleep(10)
        dfAway = self.getStats()
        self.driver.close()

        return dfHome, dfAway

    def combineData(self, dfHome, dfAway):
        combinedStats = []
        for index, row in dfHome.iterrows():
            for index2, row2 in dfAway.iterrows():
                row = row.drop(labels = ['W', 'L', 'WIN%'])
                row2 = row2.drop(labels = ['W', 'L', 'WIN%'])
                rowStats = row.to_list() + row2.to_list()
                
            combinedStats.append(rowStats)

        df = pd.DataFrame(combinedStats, columns=self.columns)
        df = df.astype(float)
        print(df)
        return df

    def predict(self, df, modelSel):
        model = Model()
        if modelSel.lower() == 'svm':
            model.loadModel('ML/models/SVM.sav')
        elif modelSel.lower() == 'knn':
            model.loadModel('ML/models/KNN.sav')
        elif modelSel.lower() == 'logistic':
            model.loadModel('ML/models/logisticRegression.sav')
        
        predictions = model.model.predict(df)
        return predictions
    
    def printResults(self, predictions, homeTeam, awayTeam):
        if predictions[0] == 1:
            print('{} will win'.format(homeTeam))
        elif predictions[0] == 0:
            print('{} will win'.format(awayTeam))
        else:
            print('Error getting prediction: {}'.format(predictions))

if __name__ == '__main__':
    predictor = Predictor()
    homeTeam, awayTeam, modelType = predictor.getUserInput()
    dfHome, dfAway = predictor.scrapeTeams(homeTeam, awayTeam)
    df = predictor.combineData(dfHome, dfAway)
    predictions = predictor.predict(df, modelType)
    predictor.printResults(predictions, homeTeam, awayTeam)
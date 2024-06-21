from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd

class WebScraper:
    driver = None
    teamIDs = {'ATL': '1610612737', 'BOS': '1610612738', 'BKN': '1610612751', 'CHA': '1610612766',
            'CHI': '1610612741', 'CLE': '1610612739', 'DAL': '1610612742', 'DEN': '1610612743',
            'DET': '1610612765', 'GSW': '1610612744', 'HOU': '1610612745', 'IND': '1610612754',
            'LAC': '1610612746', 'LAL': '1610612747', 'MEM': '1610612763', 'MIA': '1610612748',
            'MIL': '1610612749', 'MIN': '1610612750', 'NOP': '1610612740', 'NYK': '1610612752',
            'OKC': '1610612760', 'ORL': '1610612753', 'PHI': '1610612755', 'PHX': '1610612756',
            'POR': '1610612757','SAC': '1610612758', 'SAS': '1610612759', 'TOR': '1610612761',
            'UTA': '1610612762', 'WAS': '1610612764'}
    seasonYears = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
    seasonTypes = ['Regular+Season', 'Playoffs']

    def __init__(self, driver):
        self.driver = driver

    def goToPage(self, teamID, seasonType, seasonYear, statType):
        url = 'https://www.nba.com/stats/team/' + teamID + '/boxscores-' + statType + '?'
        url += 'SeasonType=' + seasonType + '&Season=' + seasonYear
        driver.get(url)

    def getTableColumns(self):
        tableHead = ["Date", "Home", "Away"]
        tableHead_xpath = '//div[3]/table/thead//th'
        NUM_COLUMNS = len(driver.find_elements(By.XPATH, tableHead_xpath))

        for i in range(1, NUM_COLUMNS):
            statName_xpath = tableHead_xpath + '[' + str(i + 1) + ']'
            statName = driver.find_element(By.XPATH, statName_xpath)
            tableHead.append(statName.text)

        return tableHead

    def getTableStats(self):
        table = []
        tableRow_xpath = '//div[3]/table/tbody/tr'
        tableRow = driver.find_elements(By.XPATH, tableRow_xpath)
        tableHead_xpath = '//div[3]/table/thead//th'
        NUM_COLUMNS = len(driver.find_elements(By.XPATH, tableHead_xpath))

        # Loop through rows
        for i in range(len(tableRow)):
            row_xpath = tableRow_xpath + '[' + str(i + 1) + ']'
            rowStats = []
            matchup = driver.find_element(By.XPATH, row_xpath + '/td[1]').text
            print(matchup) # For tracking progress

            # Split up matchup column
            temp = matchup.split(' - ')
            rowStats.append(temp[0])
            if '@' in temp[1]:
                temp = temp[1].split(' @ ')
                rowStats.append(temp[1])
                rowStats.append(temp[0])
            elif 'vs.' in temp[1]:
                temp = temp[1].split(' vs. ')
                rowStats.append(temp[0])
                rowStats.append(temp[1])

            # Loop through columns 
            for j in range(1, NUM_COLUMNS):
                stat_xpath = row_xpath + '/td[' + str(j + 1) + ']'
                stat = driver.find_element(By.XPATH, stat_xpath)
                rowStats.append(stat.text)

            table.append(rowStats)

        return table
    
    def scrapePage(self, team, seasonType, seasonYear, statType):
        self.goToPage(self.teamIDs[team], seasonType, seasonYear, statType)
        columns = self.getTableColumns()
        table = self.getTableStats()
        df = pd.DataFrame(table, columns=columns)
        fileName = team + ' ' + seasonYear + '.csv'
        df.to_csv(fileName, index=False)
        return df

    def scrapeAllTeams(self, seasonYear):
        df = pd.DataFrame()
        for team in self.teamIDs:
            temp_df = self.scrapePage(team, self.seasonTypes[0], seasonYear, 'traditional')
            df = pd.concat([df, temp_df], axis=0)

        fileName = seasonYear + '.csv'
        df.to_csv(fileName, index=False)

if __name__ == "__main__":

    driver = webdriver.Chrome()
    thing = WebScraper(driver)
    thing.scrapeAllTeams('2023-24')
    driver.close()
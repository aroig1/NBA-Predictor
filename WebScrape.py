from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
import pandas as pd
import time

class WebScraper:
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
    seasonYears = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
    seasonTypes = ['Regular+Season', 'Playoffs']

    def __init__(self):
        self.options = webdriver.ChromeOptions()
        # self.options.add_argument("--headless=new") # Run without UI popup (DOESNT WORK)
        self.options.add_argument("--ignore-certificate-errors")
        self.options.add_argument("--log-level=3")  # Suppresses most logs
        self.options.add_argument("--disable-logging")

    def goToPage(self, teamID, seasonType, seasonYear, statType):
        try:
            url = 'https://www.nba.com/stats/team/' + teamID + '/boxscores-' + statType + '?'
            url += 'SeasonType=' + seasonType + '&Season=' + seasonYear
            self.driver.get(url)
        except Exception as e:
            print(f'Exception occured while changing pages \n{e}')

    def getTableColumns(self):
        # tableHead = ["Date", "Home", "Away", "Winner (H/A)"]
        tableHead = []
        tableHead_xpath = '//div[3]/table/thead//th'

        try:
            NUM_COLUMNS = len(self.driver.find_elements(By.XPATH, tableHead_xpath))

            # Loop through columns
            for i in range(NUM_COLUMNS):
                statName_xpath = tableHead_xpath + '[' + str(i + 1) + ']'
                statName = self.driver.find_element(By.XPATH, statName_xpath)
                tableHead.append(statName.text)
        except Exception as e:
            print(f'Error while getting table column names \n{e}')

        return tableHead

    def getTableStats(self):
        table = []
        tableRow_xpath = '//div[3]/table/tbody/tr'
        tableRow = self.driver.find_elements(By.XPATH, tableRow_xpath)
        tableHead_xpath = '//div[3]/table/thead//th'
        NUM_COLUMNS = len(self.driver.find_elements(By.XPATH, tableHead_xpath))

        # Loop through rows
        for i in range(len(tableRow)):
            try:
                row_xpath = tableRow_xpath + '[' + str(i + 1) + ']'
                rowStats = []

                # Loop through columns
                for j in range(NUM_COLUMNS):
                    stat_xpath = row_xpath + '/td[' + str(j + 1) + ']'
                    stat = self.driver.find_element(By.XPATH, stat_xpath)
                    rowStats.append(stat.text)

                table.append(rowStats)
            except Exception as e:
                print(f'Error scraping row {i} \n{e}')

        return table
    
    def scrapePage(self, team, seasonType, seasonYear, statType):
        self.driver = webdriver.Chrome(options=self.options)

        self.goToPage(self.teamIDs[team], seasonType, seasonYear, statType)
        time.sleep(10)

        select = Select(self.driver.find_element(By.XPATH, '//section[3]//div[2]//select'))
        select.select_by_visible_text('All')
        time.sleep(3)

        columns = self.getTableColumns()
        table = self.getTableStats()
        df = pd.DataFrame(table, columns=columns)
        fileName = 'rawData/' + seasonYear + '/' + team + ' ' + seasonYear + '.csv'
        df.to_csv(fileName, index=False)

        self.driver.close()

        return df

    def scrapeAllTeams(self, seasonYear):
        df = pd.DataFrame()
        for team in self.teamIDs:
            try:
                print(f'Starting {team} {seasonYear}')
                temp_df = self.scrapePage(team, self.seasonTypes[0], seasonYear, 'traditional')
                df = pd.concat([df, temp_df], axis=0)
                print(f'Completed {team} {seasonYear}')
            except Exception as e:
                print(f'Error scraping team: {team} {seasonYear} \n{e}')

        fileName = 'rawData/'  + seasonYear + '/AllTeams ' + seasonYear + '.csv'
        df.to_csv(fileName, index=False)

    def scrapeAllYears(self):
        for year in self.seasonYears:
            self.scrapeAllTeams(year)

if __name__ == "__main__":

    thing = WebScraper()
    # thing.scrapeAllYears()
    # thing.scrapeAllTeams('2019-20')
    thing.scrapePage('LAL', 'Regular+Season', '2019-20', 'traditional')

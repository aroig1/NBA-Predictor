from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd


if __name__ == "__main__":

    ROW_LENGTH = 22

    driver = webdriver.Chrome()
    url = 'https://www.nba.com/stats/team/1610612737/boxscores-traditional?SeasonType=Regular+Season&Season=2023-24'
    driver.get(url)

    tableHead = ["Date", "Home", "Away"]

    tableHead_xpath = '//div[3]/table/thead//th'
    ROW_LENGTH = len(driver.find_elements(By.XPATH, tableHead_xpath))
    for i in range(1, ROW_LENGTH):
        statName_xpath = tableHead_xpath + '[' + str(i + 1) + ']'
        statName = driver.find_element(By.XPATH, statName_xpath)
        tableHead.append(statName.text)

    tableRow_xpath = '//div[3]/table/tbody/tr'
    tableRow = driver.find_elements(By.XPATH, tableRow_xpath)

    table = []

    for i in range(len(tableRow)):
        row_xpath = tableRow_xpath + '[' + str(i + 1) + ']'
        rowStats = []
        matchup = driver.find_element(By.XPATH, row_xpath + '/td[1]').text
        print(matchup) # For tracking progress
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
        for j in range(1, ROW_LENGTH):
            stat_xpath = row_xpath + '/td[' + str(j + 1) + ']'
            stat = driver.find_element(By.XPATH, stat_xpath)
            rowStats.append(stat.text)
        table.append(rowStats)

    driver.close()

    df = pd.DataFrame(table, columns=tableHead)
    df.to_csv('Hawks 2023-24.csv', index=False)
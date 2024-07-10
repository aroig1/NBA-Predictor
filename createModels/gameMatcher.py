import pandas as pd

class GameMatcher:
    seasonYears = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
    columns = ['DATE', 'HOME TEAM', 'AWAY TEAM', 'Winner (H/A)', 'HOME PTS', 
               'HOME FGM', 'HOME FGA', 'HOME FG%', 'HOME 3PM', 'HOME 3PA', 'HOME 3P%', 
               'HOME FTM', 'HOME FTA', 'HOME FT%', 'HOME OREB', 'HOME DREB', 'HOME REB', 
               'HOME AST', 'HOME TOV', 'HOME STL' ,'HOME BLK', 'HOME PF', 'HOME +/-',
               'AWAY PTS', 'AWAY FGM', 'AWAY FGA', 'AWAY FG%', 'AWAY 3PM', 'AWAY 3PA', 
               'AWAY 3P%', 'AWAY FTM', 'AWAY FTA', 'AWAY FT%', 'AWAY OREB', 'AWAY DREB', 
               'AWAY REB', 'AWAY AST', 'AWAY TOV', 'AWAY STL' ,'AWAY BLK', 'AWAY PF', 
               'AWAY +/-']

    def matchGames(self, seasonYear):
        fileName = 'averagedData/' + seasonYear + '/AllTeams ' + seasonYear + '.csv'
        df = pd.read_csv(fileName)
        matchedStats = []

        for index, row in df.iterrows():
            opp = row['OPP']
            temp_df = df[df['DATE'] == row['DATE']]
            for index2, row2 in temp_df.iterrows():
                if opp == row2['TEAM']:
                    winner = None
                    rowStats = []
                    if row['H/A'] == 'H':
                        if row['W/L'] == 'W':
                            winner = 'H'
                        else:
                            winner = 'A'
                        row = row.drop(labels = ['H/A', 'W/L'])
                        row2 = row2.drop(labels = ['DATE', 'TEAM', 'OPP', 'H/A', 'W/L'])
                        rowStats = row.to_list() + row2.to_list()
                    else:
                        if row['W/L'] == 'W':
                            winner = 'A'
                        else:
                            winner = 'H'
                        row = row.drop(labels = ['DATE', 'TEAM', 'OPP', 'H/A', 'W/L'])
                        row2 = row2.drop(labels = ['H/A', 'W/L'])
                        rowStats = row2.to_list() + row.to_list()
                    rowStats.insert(3, winner)
                    df = df.drop(index2)

            matchedStats.append(rowStats)

        df = pd.DataFrame(matchedStats, columns=self.columns)
        fileName = 'matchedData/' + seasonYear + '.csv' 
        df.to_csv(fileName, index=False)
        return df

    def matchAllYears(self):
        df = pd.DataFrame()
        for year in self.seasonYears:
            temp_df = self.matchGames(year)
            df = pd.concat([df, temp_df], axis=0)
            print(f'Completed matching games for year: {year}')

        df.to_csv('matchedData/AllYears.csv', index=False)

if __name__ == '__main__':
    thing = GameMatcher()
    # thing.matchGames('2019-20')
    thing.matchAllYears()

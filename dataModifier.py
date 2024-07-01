import pandas as pd
from team import Team

class DataModifier:
    teamNames = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND',
             'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX',
             'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
    seasonYears = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']

    def convertTeamAvgs(self, teamName, seasonYear):
        fileName = 'rawData/' + seasonYear + '/' + teamName + ' ' + seasonYear + '.csv'
        df = pd.read_csv(fileName)

        team = Team()
        columns = ['MATCH UP', 'W/L', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%',
                   'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL' ,'BLK', 'PF', '+/-']
        stats = []

        for index, row in df.iterrows():
            if index > 0:
                rowStats = team.getAvgs()
                rowStats.insert(0, row['W/L'])
                rowStats.insert(0, row['MATCH UP'])
                stats.append(rowStats)
            team.addGame(row)
        
        df = pd.DataFrame(stats, columns=columns)
        fileName = 'averagedData/' + seasonYear + '/' + teamName + ' ' + seasonYear + '.csv'
        df.to_csv(fileName, index=False)

        return df

    def convertYearAvgs(self, seasonYear):
        df = pd.DataFrame()

        print(f'Starting conversion of {seasonYear} to season averages')
        for teamName in self.teamNames:
            temp_df = self.convertTeamAvgs(teamName, seasonYear)
            df = pd.concat([df, temp_df], axis=0)
        print(f'Completed conversion of {seasonYear} to season averages')
        
        fileName = 'averagedData/' + seasonYear + '/AllTeams ' + seasonYear + '.csv'
        df.to_csv(fileName, index=False)
        return df
    
    def convertAllYearsAvgs(self):
        for year in self.seasonYears:
            self.convertYearAvgs(year)

if __name__ == "__main__":
    thing = DataModifier()
    # thing.convertTeamAvgs('ATL', '2019-20')
    # thing.convertYearAvgs('2019-20')
    thing.convertAllYearsAvgs()
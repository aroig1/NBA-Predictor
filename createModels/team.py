class Team:
    PTS = 0
    FGM = 0
    FGA = 0
    FG_percent = 0
    threePM = 0
    threePA = 0
    threeP_percent = 0
    FTM = 0
    FTA = 0
    FT_percent = 0
    OREB = 0
    DREB = 0
    REB = 0
    AST = 0
    TOV = 0
    STL = 0
    BLK = 0
    PF = 0
    plus_minus = 0
    games_played = 0

    def addGame(self, stats):
        self.PTS += stats['PTS']
        self.FGM += stats['FGM']
        self.FGA += stats['FGA']
        self.FG_percent += stats['FG%']
        self.threePM += stats['3PM']
        self.threePA += stats['3PA']
        self.threeP_percent += stats['3P%']
        self.FTM += stats['FTM']
        self.FTA += stats['FTA']
        self.FT_percent += stats['FT%']
        self.OREB += stats['OREB']
        self.DREB += stats['DREB']
        self.REB += stats['REB']
        self.AST += stats['AST']
        self.TOV += stats['TOV']
        self.STL += stats['STL']
        self.BLK += stats['BLK']
        self.PF += stats['PF']
        self.plus_minus += stats['+/-']
        self.games_played += 1

    def getAvgs(self):
        avgs = []
        avgs.append(round(self.PTS / self.games_played, 2))
        avgs.append(round(self.FGM / self.games_played, 2))
        avgs.append(round(self.FGA / self.games_played, 2))
        avgs.append(round(self.FG_percent / self.games_played, 2))
        avgs.append(round(self.threePM / self.games_played, 2))
        avgs.append(round(self.threePA / self.games_played, 2))
        avgs.append(round(self.threeP_percent / self.games_played, 2))
        avgs.append(round(self.FTM / self.games_played, 2))
        avgs.append(round(self.FTA / self.games_played, 2))
        avgs.append(round(self.FT_percent / self.games_played, 2))
        avgs.append(round(self.OREB / self.games_played, 2))
        avgs.append(round(self.DREB / self.games_played, 2))
        avgs.append(round(self.REB / self.games_played, 2))
        avgs.append(round(self.AST / self.games_played, 2))
        avgs.append(round(self.TOV / self.games_played, 2))
        avgs.append(round(self.STL / self.games_played, 2))
        avgs.append(round(self.BLK / self.games_played, 2))
        avgs.append(round(self.PF / self.games_played, 2))
        avgs.append(round(self.plus_minus / self.games_played, 2))

        return avgs

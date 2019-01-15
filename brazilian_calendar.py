import datetime as dt
from pandas.tseries.holiday import AbstractHolidayCalendar, GoodFriday, Holiday, Easter, Day
from pandas.tseries.offsets import CustomBusinessDay

class BrazilHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('Confraternização Universal', month=1, day=1),
        # Terça feira de carnaval (47 before Páscoa)
        Holiday('Segunda Feira de Carnaval', month=1, day=1, offset=[Easter(), Day(-48)]),
        Holiday('Terça Feira de Carnaval', month=1, day=1, offset=[Easter(), Day(-47)]),
        # Sexta feira Santa(GoodFriday)
        GoodFriday,        
        Holiday('Tiradentes', month = 4, day = 21),
        Holiday('Dia do Trabalho', month = 5, day = 1),
        # Corpus Christi (60 dias after Páscoa, or 62 day before Sexta Feira Santa)
        Holiday('Corpus Christi', month=1, day=1, offset=[Easter(), Day(60)]),        
        Holiday('Independência do Brasil', month = 9, day = 7),
        Holiday('Nossa Senhora Aparecida - Padroeira do Brasil', month = 10, day = 12),
        Holiday('Finados', month = 11, day = 2),
        Holiday('Proclamação da República', month = 11, day = 15),
        Holiday('Natal', month = 12, day = 25)]

# Generates holydays by data range ex: (dt.datetime(2000, 12, 31), dt.datetime(2079, 12, 31))
def get_holidays(dt_start, dt_end):
    br_holidays = CustomBusinessDay(calendar=BrazilHolidays())    
    inst = BrazilHolidays()
    return inst.holidays(dt_start, dt_end)


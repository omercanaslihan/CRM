##BG-NBD ve Gamma-Gamma ile CLTV Tahmini##
#İş Problemi#
#İngiltere merkezli perakende şirketi satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
#Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel
#değerin tahmin edilmesi gerekmektedir.

#Veri Seti Hikayesi
#Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri
#arasındaki online satış işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu
#müşterisinin toptancı olduğu bilgisi mevcuttur.

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter, GammaGammaFitter
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

#Görev 1: BG-NBD ve Gamma-Gamma Modellerini Kurarak 6 Aylık CLTV Tahmini Yapılması
#Adım 1: 2010-2011 yıllarındaki veriyi kullanarak İngiltere’deki müşteriler için 6 aylık CLTV tahmini yapınız.
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.isnull().sum()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df.describe().T

def outlier_tresholds(dataframe, variable):
    quartile_1 = dataframe[variable].quantile(0.01)
    quartile_3 = dataframe[variable].quantile(0.99)
    interquartile = quartile_3 - quartile_1
    low_limit = quartile_1 - 1.5 * interquartile
    up_limit = quartile_3 + 1.5 * interquartile
    return low_limit, up_limit

def replace_tresholds(dataframe, variable):
    low_limit, up_limit = outlier_tresholds(dataframe, variable)
    dataframe[dataframe[variable] > up_limit] = up_limit
    dataframe[dataframe[variable] < low_limit] = low_limit

replace_tresholds(df, "Quantity")
replace_tresholds(df, "Price")
df.describe().T

df["total_price"] = df["Quantity"] * df["Price"]
df.info()
today_date = dt.datetime(2011, 12, 11)
df["InvoiceDate"] = df["InvoiceDate"].apply(pd.to_datetime)
cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max()-x.min()).days,
                                        lambda x: (today_date-x.min()).days],
                                        "Invoice": lambda x: x.nunique(),
                                        "Price": lambda x: x.sum()})
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df["recency"] = cltv_df['recency'] / 7
cltv_df["T"] = cltv_df['T'] / 7
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df = cltv_df.reset_index()

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])
cltv_df["exp_purch_6_months"] = bgf.predict(4*6,
                                            cltv_df["frequency"],
                                            cltv_df["recency"],
                                            cltv_df["T"]
                                                        )
ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary"])
cltv_df["exp_ave_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                         cltv_df["monetary"])

cltv_6 = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01
                                   )
cltv_df["cltv_6"] = cltv_6
cltv_df["country"] = df["Country"]
cltv_df = cltv_df[cltv_df['country'] == "United Kingdom"]

#Görev 2: Farklı Zaman Periyotlarından Oluşan CLTV Analizi
#Adım 1: 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])
cltv_df["exp_purch_1_months"] = bgf.predict(4,
                                            cltv_df["frequency"],
                                            cltv_df["recency"],
                                            cltv_df["T"]
                                                        )
ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary"])
cltv_df["exp_ave_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                         cltv_df["monetary"])

cltv_1 = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01
                                   )
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])
cltv_df["exp_purch_12_months"] = bgf.predict(4*12,
                                            cltv_df["frequency"],
                                            cltv_df["recency"],
                                            cltv_df["T"]
                                                        )
ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary"])
cltv_df["exp_ave_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                         cltv_df["monetary"])
cltv_12 = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01
                                   )
cltv_df["cltv_1"] = cltv_1
cltv_df["cltv_12"] = cltv_12
cltv_df["country"] = df["Country"]
cltv_df = cltv_df[cltv_df['country'] == "United Kingdom"]

#Adım 2: 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
cltv_df.sort_values("cltv_1", ascending=False).head(10)
cltv_df.sort_values("cltv_12", ascending=False).head(10)

#Görev 3: Segmentasyon ve Aksiyon Önerileri
#Adım 1: 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine
#ekleyiniz.
cltv_df["segment"] = pd.qcut(cltv_df["cltv_6"], 4, labels=["D", "C", "B", "A"])

#Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
cltv_df[["segment", 'cltv_6']].groupby("segment").agg(["mean", "count"])
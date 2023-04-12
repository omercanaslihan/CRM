####BG-NBD ve Gamma-Gamma ile CLTV Tahmini#####

##İş Problemi##
#FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
#Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte
#şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

#Veri Seti Hikayesi#
#Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel
#(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş
#davranışlarından elde edilen bilgilerden oluşmaktadır.
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter, GammaGammaFitter
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.width", 1000)

#Görev 1: Veriyi Hazırlama
#Adım1: flo_data_20K.csvverisiniokuyunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()

#Adım 2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
#Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_tresholds(dataframe, variable):
    quartile_1 = dataframe[variable].quantile(0.01)
    quartile_3 = dataframe[variable].quantile(0.99)
    interquartlile = quartile_3 - quartile_1
    low_limit = round(quartile_1 - 1.5 * interquartlile, 0)
    up_limit = round(quartile_3 + 1.5 * interquartlile, 0)
    return low_limit, up_limit

def replace_with_treshold(dataframe, variable):
    low_limit, up_limit = outlier_tresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit

#Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
#"customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.
cols = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in cols:
    replace_with_treshold(df, col)

#Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
#Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["total_orders"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["total_customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

#Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

#Görev 2: CLTV Veri Yapısının Oluşturulması
#Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

#Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
#Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["frequency"] = df["total_orders"]
cltv_df["monetary_cltv_avg"] = df["total_customer_value"] / df["total_orders"]
cltv_df.head()

#Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
#Adım 1: BG/NBD modelini fit ediniz.
#• 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
#• 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                        cltv_df["frequency"],
                                        cltv_df["recency_cltv_weekly"],
                                        cltv_df["T_weekly"])
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                        cltv_df["frequency"],
                                        cltv_df["recency_cltv_weekly"],
                                        cltv_df["T_weekly"])
cltv_df.head()

#Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])
cltv_df.head()

#Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz. • Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01
                                   )
cltv_df["cltv"] = cltv
cltv_df.sort_values("cltv", ascending=False)[:20]

#Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
#Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()



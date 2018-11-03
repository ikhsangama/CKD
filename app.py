from flask import Flask, render_template, request
import pymysql
import numpy as np

import MyDir as my

connection = pymysql.connect(host="localhost", user="root", passwd="", database="knn")
cursor = connection.cursor()

app = Flask(__name__)

# ######################################################
import numpy as np
import pandas as pd
import pymysql as pskl
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt


# Menampilkan semua array
# np.set_printoptions(threshold=np.nan)

#
# 2
#
def dataset(retrieve="all", id=1, target=1):  # hasil bertipe dataframe
    """
    parameters
    ----------
    retrieve: (all, numeric, polinom)
        all    : semua kolom
        numeric: hanya kolom numeric
        polinom: hanya kolom polinom
    id: (0, 1)
        0: tanpa kolom id
        1: dengan kolom id
    class: (0, 1)
        0: tanpa kolom class
        1: dengan kolom class

    """
    connection = pskl.connect(host="localhost", user="root", passwd="", database="knn")
    cursor = connection.cursor()
    if (retrieve == "all"):
        cols = ["id", "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
                "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]
        retrieve = "SELECT id, age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane, class FROM ckd_preprocessing3"
    elif (retrieve == "numeric"):
        cols = ["id", "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
                "class"]
        retrieve = "SELECT id, age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, class FROM ckd_preprocessing3"
    elif (retrieve == "polinom"):
        cols = ["id", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]
        retrieve = "SELECT id, rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane, class FROM ckd_preprocessing3"
    resolveall = cursor.execute(retrieve)
    rows_tupple = cursor.fetchall()
    data = pd.DataFrame(list(rows_tupple))
    data.columns = cols
    if (id == 0):
        data = data.drop(["id"], axis=1)
    if (target == 0):
        data = data.drop(["class"], axis=1)
    data = data.fillna(value=np.nan)  # mengubah missing value menjadi NaN
    return data


#
# 4.1
#
def encoding(df, column="all_nominal",
             all_nominal=["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]):
    """
    paramters:
    ----------
    df: input tipe dataframe
    column: input tipe string, menentukan kolom nominal mana yang akan di transformasi menjadi numerik

    output:
    ----------
    mengembalikan dataframe dengan kolom tertentu yang sudah di encode selain "NaN"

    """
    # python melakukan pass by reference, sehingga dibuat copy agar df sebelumnya tidak berubah

    copy_df = df.copy()
    if (column != "all_nominal"):
        a = copy_df[column].unique().tolist()
        c = [x for x in a if str(x) != 'nan']
        l = []
        i = 0
        for x in c:
            l.append(i)
            i += 1
        copy_df[column] = copy_df[column].replace(c, l)

    elif (column == "all_nominal"):
        #         all_nominal = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane","class"]
        for col in all_nominal:
            copy_df = encoding(copy_df, col)  # rekursif

    return copy_df


#
# 4.2
#
def missing_handling(df, column="all", method="mean",
                     all_col=["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc",
                              "rbcc", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]):
    """
    paramters:
    ----------
    df: input tipe dataframe
    column: menentukan kolom numerik mana yang akan dilakukan penanganan missing value
    method: strategi penanganan missing value

    output:
    ----------
    mengembalikan dataframe dengan kolom yang sudah dilakukan penanganan NaN
    """

    copy_df = df.copy()
    imputer = Imputer(missing_values="NaN", strategy=method, axis=0)

    if (column != "all"):
        imputer = imputer.fit(copy_df[[column]])
        filledmissing_df = imputer.transform(copy_df[[column]])
        df_change = filledmissing_df.ravel()
        copy_df[column] = df_change

    elif (column == "all"):
        #         all_col = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]
        for col in all_col:
            copy_df = missing_handling(copy_df, col, method)  # rekursif
    return copy_df


#
# 4.3 NORMALISASI
#
from sklearn.preprocessing import MinMaxScaler


def normalizing(df, column="all", f_range=(0, 1),
                all_col=["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
                         "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]):
    """
    paramters:
    ----------
    df: input tipe dataframe
    column: menentukan kolom numerik mana yang akan dilakukan penanganan missing value
    range: range normalisasi

    output:
    ----------
    mengembalikan dataframe dengan kolom yang sudah dinormalisasi
    """

    copy_df = df.copy()
    scale = MinMaxScaler(feature_range=f_range)

    if (column != "all"):
        normalization_array = scale.fit_transform(copy_df[[column]])
        df_change = normalization_array.ravel()
        copy_df[column] = df_change

    elif (column == "all"):
        #         all_col = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
        for col in all_col:
            copy_df = normalizing(copy_df, col, f_range)  # rekursif
    return copy_df


#
# 6.1 Klasifikasi dengan kNN
#
import MyDir as my

#
# 7.1 Mencari nilai terbaik k pada kNN
#
from sklearn.model_selection import KFold


def knn_kfold(df, kNN=5, k_fold=10, shuffle_=True, arr_akurasi=0):
    """
    Parameter:
    ----------
    df : Dataframe, data bertipe float / integer, tidak boleh ada NaN
    kNN : Nilai k-tetangga/data terdekat, default = 5
    k_fold : Jumlah fold / lipatan split training - test. Default = 5
    shuffle_ : Bentuk shuffle split jika diperlukan. Default = True

    Ouput:
    -------
    arr_akurasi : 1 untuk menghasilkan output array akurasi tiap fold, 0 untuk rata - rata akurasi. default = 0
    """

    kf = KFold(n_splits=k_fold, shuffle=shuffle_, random_state=2)
    akurasi = []
    for train_index, test_index in kf.split(df):
        train_df = df.iloc[train_index]
        x_train = np.array(train_df.iloc[:, 1:-1])
        y_train = np.array(train_df["class"])
        test_df = df.iloc[test_index]
        x_test = np.array(test_df.iloc[:, 1:-1])
        y_test = np.array(test_df["class"])

        my_predictions = np.array([my.knn_predict(p, x_train, y_train, k=kNN) for p in x_test])
        akurasi_my_predictions = np.mean(my_predictions == y_test) * 100
        akurasi.append(akurasi_my_predictions)
    if (arr_akurasi == 1):
        return akurasi
    else:
        return np.mean(akurasi)


def find_best_knn(df, start=3, finish=16, step=2, graph=True):
    akurasi = []
    maks = 0
    for i in range(start, finish, step):
        akurasi_k = knn_kfold(df, kNN=i, k_fold=10, shuffle_=True, arr_akurasi=0)
        akurasi.append([i, akurasi_k])
        if (maks < akurasi_k):
            maks = akurasi_k
            k = i
            print(k, maks)
        if (graph == True):
            x = [i[0] for i in akurasi]
            y = [i[1] for i in akurasi]
            plt.plot(x, y)
            plt.title("Mencari k Terbaik kNN")
            plt.xlabel("k")
            plt.ylabel("akurasi")
            plt.grid()
            plt.show
    return akurasi


import statsmodels.formula.api as sm


def backward_elimination(splitdf_x, splitdf_y, sl=0.05):
    sl_ = sl
    copy_df = splitdf_x.copy()
    reg_ols = sm.OLS(endog=splitdf_y, exog=copy_df).fit()
    pval = reg_ols.pvalues
    maks_pval = max(pval)

    if (maks_pval >= sl_):
        delete_col = pval[pval == maks_pval].index[0]
        drp = [delete_col]
        copy_df = copy_df.drop(drp, axis=1)
        copy_df = backward_elimination(copy_df, splitdf_y, sl_)

    else:
        copy_df = pd.concat([copy_df, splitdf_y], axis=1)
    return copy_df


# BIAR GA LAMA
dataset_df = dataset()
# encoded_df = encoding(dataset_df)
# missing_handling_df = missing_handling(encoded_df)
# normalized_df = normalizing(missing_handling_df)
# knn = find_best_knn(normalized_df)
# x = normalized_df.iloc[:,1:-1]
# y = normalized_df.iloc[:,25:]
# backwarded_df = backward_elimination(x,y,0.05)
# backwarded_knn = knn_kfold(backwarded_df, kNN = 5, k_fold=10, shuffle_=True, arr_akurasi=0)
# .BIAR GA LAMA
@app.route('/')
def index():
    # predictors = training()
    # outcomes = target()
    #
    # my_predictions = np.array([my.knn_predict(p, predictors, outcomes, 3) for p in predictors])
    # akurasi1 = np.mean(my_predictions == outcomes) * 100
    #
    # from sklearn.neighbors import KNeighborsClassifier
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(predictors, outcomes)
    # sk_predictions = knn.predict(predictors)
    #
    # akurasi2 = np.mean(sk_predictions == outcomes) * 100
    return render_template('index.html')


@app.route('/datamining_1.html')
def datamining_1():
    return render_template('datamining_1.html')


@app.route('/datamining_2.html')
def datamining_2():
    dataset_df = dataset()
    return render_template('datamining_2.html', dataset_df=dataset_df)


@app.route('/datamining_3.html')
def datamining_3():
    return render_template('datamining_3.html')


@app.route('/datamining_4.html')
def datamining_4():
    return render_template('datamining_4.html', encoded_df=encoded_df, missing_handling_df=missing_handling_df)


@app.route('/datamining_5.html')
def datamining_5():
    return render_template('datamining_5.html', encoded_df=encoded_df, missing_handling_df=missing_handling_df)


@app.route('/identifikasi.html')
def identifikasi():
    return render_template('identifikasi.html')


@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        nama = request.form['nama']
        l1 = []
        if (request.form['sg'] != ""):
            p1 = float(request.form['sg'])
        else:
            p1 = "NaN"
        if (request.form['al'] != ""):
            p2 = float(request.form['sg'])
        else:
            p2 = "NaN"
        if (request.form['bu'] != ""):
            p3 = float(request.form['bu'])
        else:
            p3 = "Nan"
        if (request.form['sc'] != ""):
            p4 = float(request.form['sc'])
        else:
            p4 = "Nan"
        if (request.form['sod'] != ""):
            p5 = float(request.form['sod'])
        else:
            p5 = "NaN"
        if (request.form['hemo'] != ""):
            p6 = float(request.form['hemo'])
        else:
            p6 = "Nan"
        p7 = request.form['rbc']
        p8 = request.form['htn']
        p9 = request.form['dm']
        p10 = request.form['appet']
        l1.append(p1)
        l1.append(p2)
        l1.append(p3)
        l1.append(p4)
        l1.append(p5)
        l1.append(p6)
        l1.append(p7)
        l1.append(p8)
        l1.append(p9)
        l1.append(p10)

        # MENGAMBIL INPUT SEBAGAI DF DAN MENGGABUNGKAN DENGAN TRAINING DF ATRIBUT TERBAIK
        col = ['sg', 'al', 'bu', 'sc', 'sod', 'hemo', 'rbc', 'htn', 'dm', 'appet']
        array_p = np.array(l1)
        p_df = pd.DataFrame(array_p.reshape(1, -1), columns=col)
        dataset_df = dataset()
        cut_df = dataset_df[col]
        inputed_df = cut_df.append(p_df, ignore_index=True)
        # .MENGAMBIL INPUT SEBAGAI DF DAN MENGGABUNGKAN DENGAN TRAINING DF ATRIBUT TERBAIK
        # TRANSFORMASI & NORMALISASI INPUT
        new_point, points = best_attribute_training(inputed_df)
        # .TRANSFORMASI & NORMALISASI INPUT
        outcomes = target()
        print(new_point)
        print(points)
        # points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
        # outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        hasil = my.knn_predict(new_point, points, outcomes)
        if (hasil == "ckd"):
            hasil = "positif ginjal kronis"
        else:
            hasil = "negatif ginjal kronis"
    return render_template('output.html', hasil=hasil, nama=nama, list=l1)


def training():
    # retrieve = "Select rbcc_2, sg_2, al_2, rbc_2, pot_2, htn_2, dm_2, pc_2, pe_2, ane_2, pcc_2 from ckd_preprocessing"
    retrieve = "Select sg_2, al_2, dm_2, htn_2,hemo_2, sc_2, rbcc_2, pcc_2, appet_2, sod_2 from ckd_preprocessing"
    # retrieve = "Select sg, al, dm_1, htn_1, hemo, sc, rbcc, pcc_1, appet_1, sod from ckd_preprocessing"
    cursor.execute(retrieve)
    rows = cursor.fetchall()
    p = np.array([list(row) for row in rows])
    w = p.astype(float)
    return w


def best_attribute_training(dataset):
    # return pertama yang akan diuji, return kedua data training
    col = ['sg', 'al', 'bu', 'sc', 'sod', 'hemo', 'rbc', 'htn', 'dm', 'appet']
    df = dataset
    cut_df = df[col]
    col_nominal = ["rbc", "htn", "dm", "appet"]
    encoding_df = encoding(cut_df, all_nominal=col_nominal)
    missing_handling_df = missing_handling(encoding_df, all_col=col)
    normalizing_df = normalizing(missing_handling_df, all_col=col)
    return normalizing_df.values[-1].tolist(), normalizing_df.iloc[:-1].values


def target():
    target = "Select class from ckd_preprocessing"
    cursor.execute(target)
    ha = cursor.fetchall()
    outcomes = []
    for row in ha:
        for i in row:
            outcomes.append(i)
    outcomes = np.array(outcomes)
    return outcomes


@app.route('/demo.html')
def demo1():
    return render_template("demo.html")


@app.route('/demo_1send', methods=['GET', 'POST'])
def demo_1send():
    terbaik = False
    if request.method == "POST":
        if (request.form['age'] != ""):
            age = float(request.form['age'])
        else:
            age = "NaN"
        if (request.form['bp'] != ""):
            bp = float(request.form['bp'])
        else:
            bp = "NaN"
        if (request.form['sg'] != ""):
            sg = float(request.form['sg'])
        else:
            sg = "NaN"
        if (request.form['al'] != ""):
            al = float(request.form['al'])
        else:
            al = "NaN"
        if (request.form['su'] != ""):
            su = float(request.form['su'])
        else:
            su = "Nan"
        if (request.form['bgr'] != ""):
            bgr = float(request.form['bgr'])
        else:
            bgr = "Nan"
        if (request.form['bu'] != ""):
            bu = float(request.form['bu'])
        else:
            bu = "NaN"
        if (request.form['sc'] != ""):
            sc = float(request.form['sc'])
        else:
            sc = "Nan"
        if (request.form['sod'] != ""):
            sod = float(request.form['sod'])
        else:
            sod = "Nan"
        if (request.form['pot'] != ""):
            pot = float(request.form['pot'])
        else:
            pot = "Nan"
        if (request.form['hemo'] != ""):
            hemo = float(request.form['hemo'])
        else:
            hemo = "Nan"
        if (request.form['pcv'] != ""):
            pcv = float(request.form['pcv'])
        else:
            pcv = "Nan"
        if (request.form['wbcc'] != ""):
            wbcc = float(request.form['wbcc'])
        else:
            wbcc = "Nan"
        if (request.form['rbcc'] != ""):
            rbcc = float(request.form['rbcc'])
        else:
            rbcc = "Nan"
        rbc = request.form['rbc']
        pc = request.form['pc']
        pcc = request.form['pcc']
        ba = request.form['ba']
        htn = request.form['htn']
        dm = request.form['dm']
        cad = request.form['cad']
        appet = request.form['appet']
        pe = request.form['pe']
        ane = request.form['ane']

        if request.form.get('terbaik', False):
            terbaik = True
            l1 = [sg, al, bu, sc, sod, hemo, rbc, htn, dm, appet]
            array_p = np.array(l1)
            col = ['sg', 'al', 'bu', 'sc', 'sod', 'hemo', 'rbc', 'htn', 'dm', 'appet']
            # p_df = pd.DataFrame(array_p.reshape(1, -1), columns=col).values
        else:
            l1 = [age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, rbc, pc, pcc, ba, htn, dm, cad,
                  appet, pe, ane]
            array_p = np.array(l1)

        if terbaik==True:
            df = dataset_df[col].values
        else:
            df = dataset_df.values

    return render_template("demo_2.html", new_point=array_p, terbaik=terbaik, df = df)


if __name__ == '__main__':
    app.debug = True
    app.run()

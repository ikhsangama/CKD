import os

from flask import Flask, render_template, request, redirect
import pymysql

import MyDir as my

connection = pymysql.connect(host="localhost", user="root", passwd="", database="knn")
cursor = connection.cursor()
# awalnya bagian demo bisa menerima semua atribut dan data yang tidak diisi diubah jadi NaN
# tapi kalau seperti itu, bagian
app = Flask(__name__)

# ######################################################
import numpy as np
import pandas as pd
import pymysql as pskl
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import time

connection = pskl.connect(host="localhost", user="root", passwd="", database="knn")
cursor = connection.cursor()


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
    # connection = pskl.connect(host="localhost", user="root", passwd="", database="knn")
    # cursor = connection.cursor()
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
# 3
#
def outliers(df, k1=0.25, k3=0.75, multiply=1.5):
    """
    parameters:
    -----------
    df: input tipe dataframe, hanya menerima numeric

    mendeteksi data yang diluar batas bawah dan batas atas
    batas bawah = k1 - (k3-k1)*1.5
    batas atas = k3 + (k3-k1)*1.5
    """
    pencilan = df.apply(
        lambda x: (x < df[x.name].quantile(k1) - ((df[x.name].quantile(k3) - df[x.name].quantile(k1)) * 1.5)) | (
                x > df[x.name].quantile(k3) + ((df[x.name].quantile(k3) - df[x.name].quantile(k1)) * 1.5)), axis=0)
    return pencilan


def outliers_removing(df, pencilan):
    """
    paramters:
    ----------
    df: input tipe dataframe
    pencilan: input tipe dataframe, bertipe boolean

    output:
    -------
    mengembalikan dataframe yang sudah menghilangkan semua outliers (semua baris yang berisi nilai pencilan true)
    """
    filtered_df = df[~(pencilan).any(axis=1)]
    return filtered_df


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

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


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
    precision = []
    recall = []
    confusion = []
    for train_index, test_index in kf.split(df):
        train_df = df.iloc[train_index]
        x_train = np.array(train_df.iloc[:, 1:-1])  # dimulai dari tanpa kolom id
        y_train = np.array(train_df["class"])
        test_df = df.iloc[test_index]
        x_test = np.array(test_df.iloc[:, 1:-1])  # dimulai dari tanpa kolom id
        y_test = np.array(test_df["class"])

        my_predictions = np.array([my.knn_predict(p, x_train, y_train, k=kNN) for p in x_test])
        #         confusion matrix
        tp, fn, fp, tn = confusion_matrix(y_test, my_predictions).ravel()
        confusion.append([tp, fn, fp, tn])
        akurasi.append((tp + tn) / (tp + tn + fp + fn) * 100)
        precision.append(((tp) / (tp + fp)) * 100)
        recall.append(((tp) / (tp + fn)) * 100)
    #         .confusion matrix

    if (arr_akurasi == 1):
        return confusion, akurasi, precision, recall
    else:
        return np.mean(akurasi), np.mean(precision), np.mean(recall)


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


# CLASS PUBLIC DATASTORE
class DataStore():
    terbaik = False
    dataset_df = dataset()
    target_df = dataset_df['class']
    new_point_awal = np.empty(0)  # inputan user
    new_point = np.empty(0)
    dict_tahap_demo = {}
    col = []
    # SEMUA ATRIBUT


data = DataStore()


# .CLASS PUBLIC DATASTORE

@app.route('/')
def index():
    rule = request.url_rule.rule
    class_counts = data.dataset_df["class"].value_counts()
    return render_template('index.html', class_counts=class_counts, rule=rule)


@app.route('/deteksi.html')
def identifikasi():
    return render_template('deteksi.html')


@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        nama = request.form['nama']
        l1 = []
        if request.form['sg'] != "":
            p1 = float(request.form['sg'])
        else:
            p1 = "NaN"
        if request.form['al'] != "":
            p2 = float(request.form['sg'])
        else:
            p2 = "NaN"
        if request.form['bu'] != "":
            p3 = float(request.form['bu'])
        else:
            p3 = "NaN"
        if request.form['sc'] != "":
            p4 = float(request.form['sc'])
        else:
            p4 = "Nan"
        if request.form['sod'] != "":
            p5 = float(request.form['sod'])
        else:
            p5 = "NaN"
        if request.form['hemo'] != "":
            p6 = float(request.form['hemo'])
        else:
            p6 = "NaN"
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


def best_attribute_training(dataset, col=['sg', 'al', 'bu', 'sc', 'sod', 'hemo', 'rbc', 'htn', 'dm', 'appet']):
    # return pertama yang akan diuji, return kedua data training
    # col = ['sg', 'al', 'bu', 'sc', 'sod', 'hemo', 'rbc', 'htn', 'dm', 'appet']
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


@app.route('/pemodelan_langsung')
@app.route('/pemodelan_manual')
def pemodelan_manual():
    rule = request.url_rule.rule
    return render_template("pemodelan_pertahap.html", rule=rule)


@app.route('/manual1', methods=['GET', 'POST'])
def render_1():
    data.dataset_df = dataset()
    data.dict_tahap_demo = {}
    data.col = []
    data.dataset_df = demo_1send()
    if len(data.col) == 0:
        return redirect("pemodelan_manual")
    df_id = data.dataset_df["id"]
    df_col = data.dataset_df[data.col]
    df_class = data.dataset_df["class"]
    data.dataset_df = pd.concat([df_id, df_col, df_class], axis=1)

    return render_template("manual1.html", shape=data.dataset_df.shape, tahapan=data.dict_tahap_demo,
                           df=data.dataset_df)


def demo_1send():
    if request.method == "POST":
        if (request.form.get('age', False) != False):
            data.col.append('age')
        if (request.form.get('bp', False) != False):
            data.col.append('bp')
        if (request.form.get('sg', False) != False):
            data.col.append('sg')
        if (request.form.get('al', False) != False):
            data.col.append('al')
        if (request.form.get('su', False) != False):
            data.col.append('su')
        if (request.form.get('bgr', False) != False):
            data.col.append('bgr')
        if (request.form.get('bu', False) != False):
            data.col.append('bu')
        if (request.form.get('sc', False) != False):
            data.col.append('sc')
        if (request.form.get('sod', False) != False):
            data.col.append('sod')
        if (request.form.get('pot', False) != False):
            data.col.append('pot')
        if (request.form.get('hemo', False) != False):
            data.col.append('hemo')
        if (request.form.get('pcv', False) != False):
            data.col.append('pcv')
        if (request.form.get('wbcc', False) != False):
            data.col.append('wbcc')
        if (request.form.get('rbcc', False) != False):
            data.col.append('rbcc')

        # MULAI NOMINAL
        if (request.form.get('rbc', False) != False):
            data.col.append('rbc')
        if (request.form.get('pc', False) != False):
            data.col.append('pc')
        if (request.form.get('pcc', False) != False):
            data.col.append('pcc')
        if (request.form.get('ba', False) != False):
            data.col.append('ba')
        if (request.form.get('htn', False) != False):
            data.col.append('htn')
        if (request.form.get('dm', False) != False):
            data.col.append('dm')
        if (request.form.get('cad', False) != False):
            data.col.append('cad')
        if (request.form.get('appet', False) != False):
            data.col.append('appet')
        if (request.form.get('pe', False) != False):
            data.col.append('pe')
        if (request.form.get('ane', False) != False):
            data.col.append('ane')

        tahapan = request.form['atribut']
        if tahapan == 0:
            tahapan = "Atribut pilihan: " + ', '.join(data.col)
        elif tahapan == 1:
            tahapan = "Atribut hasil Backward Elimination, α = 0.1: " + ', '.join(data.col)
        elif tahapan == 2:
            tahapan = "Atribut hasil Backward Elimination, α = 0.05: " + ', '.join(data.col)
        elif tahapan == 3:
            tahapan = "Semua atribut"

        data.dict_tahap_demo['1'] = tahapan
        df = data.dataset_df

    return df


# PENANGANAN OUTLIER
@app.route('/manual2')
def render_2():
    data.dataset_df, shape = demo_2send()
    return render_template("manual2.html", terbaik=data.terbaik, df=data.dataset_df,
                           shape=shape)


def demo_2send():
    data.dict_tahap_demo['2'] = "Pembersihan Outlier"

    daftar_num_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc",
                       "rbcc"]  # untuk cek outlier
    num_cols = [i for i in data.col if i in daftar_num_cols]
    numeric_df = data.dataset_df[num_cols]
    pencilan_df = outliers(numeric_df)
    filtered_df = outliers_removing(data.dataset_df, pencilan_df)
    dataset_df = filtered_df
    shape = dataset_df.shape
    return dataset_df, shape


# .PENANGANAN OUTLIER


# NOMINAL -> NUMERIK
@app.route('/manual3')
def render_3():
    data.dataset_df = demo_3send()
    return render_template("manual3.html", terbaik=data.terbaik, df=data.dataset_df,
                           shape=data.dataset_df.shape)


def demo_3send():
    data.dict_tahap_demo['3'] = "Transformasi Nominal Menjadi Numerik"

    daftar_col_nominal = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]  # untuk transformasi

    col_nominal = [i for i in data.col if i in daftar_col_nominal]

    cut_df_id = data.dataset_df["id"]
    cut_df_col = data.dataset_df[data.col]
    cut_df_class = data.dataset_df["class"]
    cut_df = pd.concat([cut_df_id, cut_df_col, cut_df_class], axis=1)

    encoding_df = encoding(cut_df, all_nominal=col_nominal)
    return encoding_df


# .NOMINAL -> NUMERIK


# MISSING HANDLING
@app.route('/manual4')
def render_4():
    data.dataset_df = demo_4send()
    return render_template("manual4.html", terbaik=data.terbaik, df=data.dataset_df, shape=data.dataset_df.shape)


def demo_4send():
    data.dict_tahap_demo['4'] = "Penanganan Missing Value"
    missing_handling_df = missing_handling(data.dataset_df, all_col=data.col)
    return missing_handling_df


# .MISSING HANDLING


# NORMALISASI
@app.route('/manual5')
def render_5():
    data.dict_tahap_demo['5'] = "Normalisasi"
    normalizing_df = normalizing(data.dataset_df, all_col=data.col)
    data.dataset_df = normalizing_df
    return render_template("manual5.html", terbaik=data.terbaik, df=data.dataset_df, shape=data.dataset_df.shape)


# .NORMALISASI -> RATA2


# PREDIKSI kNN
@app.route('/manual6', methods=['GET', 'POST'])
def render_6():
    kNN = demo_6send()
    data.dict_tahap_demo['6'] = "Prediksi kNN, k= " + str(kNN)

    start_time = time.time()
    confusion, akurasi, precision, recall = knn_kfold(data.dataset_df, arr_akurasi=1, kNN=kNN)
    waktu_validasi = time.time() - start_time

    m_akurasi = round(np.mean(akurasi), 3)
    m_precision = round(np.mean(precision), 3)
    m_recall = round(np.mean(recall), 3)

    return render_template("manual6.html", tahap=data.dict_tahap_demo,
                           terbaik=data.terbaik, df=data.dataset_df, shape=data.dataset_df.shape,
                           waktu_validasi=waktu_validasi,
                           akurasi=akurasi, precision=precision, recall=recall, m_akurasi=m_akurasi,
                           m_precision=m_precision,
                           m_recall=m_recall)


def demo_6send():
    if request.method == "POST":
        kNN = int(request.form['kNN'])

    return kNN


# .PREDIKSI kNN

if __name__ == '__main__':
    app.debug = True
    app.run()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import statsmodels
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from reportlab.platypus import PageBreak, Spacer, SimpleDocTemplate, Table, TableStyle, Preformatted, Paragraph, Frame, PageBegin, tables
from reportlab.lib import colors
import seaborn as sns
import datetime
import random
from scipy.stats.stats import pearsonr
import math
import sklearn.preprocessing
import os
import xlsxwriter
class dataVisualizer(object):
    def __init__(self, fileName1, fileName2):
        self.initializeDatasets(fileName1= fileName1, fileName2= fileName2)

    def initializeDatasets(self, fileName1, fileName2):
        #Read the data from the csv Files through Pandas
        self.dataSet1 = pd.DataFrame(pd.read_csv(fileName1))
        self.dataSet2 = pd.DataFrame(pd.read_csv(fileName2))
        #Set the variables for Bid values and ask values from both files.
        self.bidValues1 = self.dataSet1["Bid"]
        self.bidValue2 = self.dataSet2["Bid"]
        self.askValue1 = self.dataSet1["Ask"]
        self.askValue2 = self.dataSet2["Ask"]

    def splitHours(self):
        firstEncounter = False

        checker = 1.0
        self.listofFixedTimes1 = []
        self.listofFixedTimes2 = []

        # for i in range(len(self.floatTimes1)):
        #     if(self.floatTimes1[i] == checker):
        #         checker+=1
        #     listofFixedTimes.append(checker)
        for j in range(len(self.floatTimes1)):
            self.listofFixedTimes1.append(math.floor(self.floatTimes1[j] / 3600))

        for j in range(len(self.floatTimes2)):
            self.listofFixedTimes2.append(math.floor(self.floatTimes2[j]/ 3600))

        # with open("a.txt", "w") as filewriter:
        #     for i in range(len(self.floatTimes1)):
        #         filewriter.write(str(listofFixedTimes[i]) +"\n")
        # filewriter.close()

    def calculateBidAskAvg(self):
        self.bidAskAvg1 = []
        self.bidAskAvg2 = []
        size = 0
        for i in range(len(self.dataSet1)):
            if(self.dataSet1["Time"][i][0] == "1" and self.dataSet1["Time"][i][1] == "8"):
                size = i
                break
        for i in range(len(self.dataSet2)):
            if(self.dataSet2["Time"][i][0] == "1" and self.dataSet2["Time"][i][1] == "8"):
                if(size < i):
                    size = i
                break

        for i in range(size):
                self.bidAskAvg1.append( (self.dataSet1["Bid"][i] + self.dataSet1["Ask"][i]) / 2 )

        for i in range(size):
                self.bidAskAvg2.append( (self.dataSet2["Bid"][i] + self.dataSet2["Ask"][i]) / 2 )

    def writePDF(self):
        c = canvas.Canvas("ScriptTestingResults.pdf")
        NormalStyle = tables.TableStyle([

            ("BOX", (0, 0), (-1, -1), 0.45, colors.red),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.lightblue),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ])
        tableDate = []
        tableDate.append(((""), ("BidAskAvgReturnsOne       "), ("BidAskAvgReturnsTwo       "), ("AvgReturns")))
        tableDate.append(((("Max"),self.tableDF["Max"][0],self.tableDF["Max"][1],self.tableDF["Max"][2])))
        tableDate.append(((("Min"),self.tableDF["Min"][0], self.tableDF["Min"][1], self.tableDF["Min"][2])))
        tableDate.append(((("Mean"),self.tableDF["Mean"][0], self.tableDF["Mean"][1], self.tableDF["Mean"][2])))
        tableDate.append(((("Curr"),self.tableDF["Curr"][0], self.tableDF["Curr"][1], self.tableDF["Curr"][2])))
        tableDate.append(((("Skew"),self.tableDF["Skew"][0], self.tableDF["Skew"][1], self.tableDF["Skew"][2])))
        tableDate.append(((("kurtosis"),self.tableDF["kurtosis"][0], self.tableDF["kurtosis"][1], self.tableDF["kurtosis"][2])))
        myTable = tables.Table(tableDate, colWidths=1.5*inch, rowHeights=1.6*inch, style=NormalStyle)
        w, h = myTable.wrapOn(c, 1, 1)
        myTable.drawOn(c, 1.4*inch, .2*inch)

        c.showPage()


        c.drawImage("corrOneGroup.png", 10, 10, 700, 700)
        c.showPage()

        c.drawImage("corrTwoGroup.png", 10, 10, 700, 700)
        c.showPage()

        c.drawImage("corrOneTwo.png", 10, 10, 700, 700)
        c.showPage()

        c.drawImage("fullcorr.png", 10, 10, 700, 700)
        c.showPage()

        c.drawImage("SharpeHeatMap3.png", 10, 10, 700, 700)
        c.showPage()

        c.drawImage("GrpAskAvgReturnsOneHisto.png", 10, 10, 570, 700)
        c.showPage()

        c.drawImage("GrpAskAvgReturnsTwoHisto.png",10, 10, 570, 700)
        c.showPage()

        c.drawImage("GrpAskAvgVolatilityOneHisto.png", 10, 10, 570, 700)
        c.showPage()

        c.drawImage("GrpAskAvgVolatilityTwoHisto.png", 10, 10, 570, 700)
        c.showPage()

        c.drawImage("StackedPlot.png", 10, 10, 570, 700)
        c.showPage()

        c.save()
    def generateTaularData(self):
        df = pd.DataFrame()
        listofMax = []
        listofMax.append(np.max(self.bidAskAvgReturnsOne))
        listofMax.append(np.max(self.bidAskAvgReturnsTwo))
        listofMax.append(np.max(self.groupAvg))

        listofMin = []
        listofMin.append(np.min(self.bidAskAvgReturnsOne))
        listofMin.append(np.min(self.bidAskAvgReturnsTwo))
        listofMin.append(np.min(self.groupAvg))

        listofMean = []
        listofMean.append(np.mean(self.bidAskAvgReturnsOne))
        listofMean.append(np.mean(self.bidAskAvgReturnsTwo))
        listofMean.append(np.mean(self.groupAvg))

        df["Max"] = listofMax
        df["Min"] = listofMin
        df["Mean"] = listofMean

        listofAutoCurr = []
        listofAutoCurr.append((df["Max"].autocorr()))
        listofAutoCurr.append((df["Min"].autocorr()))
        listofAutoCurr.append((df["Mean"].autocorr()))

        listofSkew = []
        listofSkew.append(skew(self.bidAskAvgReturnsOne, bias=False))
        listofSkew.append(skew(self.bidAskAvgReturnsTwo, bias=False))
        listofSkew.append(skew(self.groupAvg, bias=False))

        listofKurtosis = []
        listofKurtosis.append(kurtosis(self.bidAskAvgReturnsOne, bias= False))
        listofKurtosis.append(kurtosis(self.bidAskAvgReturnsTwo, bias= False))
        listofKurtosis.append(kurtosis(self.groupAvg, bias=False))
        df["Curr"] = listofAutoCurr
        df["Skew"] = listofSkew
        df["kurtosis"] = listofKurtosis
        self.tableDF = df
        workBook = xlsxwriter.Workbook("TabularData.xlsx")
        workSheet = workBook.add_worksheet()
        bold = workBook.add_format({"bold" : True})
        workSheet.write(0, 1, "Max", bold)
        workSheet.write(0, 2, "Min", bold)
        workSheet.write(0, 3, "Mean", bold)
        workSheet.write(0, 4, "AutoCurr", bold)
        workSheet.write(0, 5, "Skew", bold)
        workSheet.write(0, 6, "kurtosis", bold)

        workSheet.write(1, 0, "BidAskAvgReturnsOne", bold)
        workSheet.write(2, 0, "bidAskAvgReturnsTwo", bold)
        workSheet.write(3, 0, "groupAvgReturns", bold)

        self.writer(workSheet, 1, 1, df, "Max")
        self.writer(workSheet, 1, 2, df, "Min")
        self.writer(workSheet, 1, 3, df, "Mean")
        self.writer(workSheet, 1, 4, df, "Curr")
        self.writer(workSheet, 1, 5, df, "Skew")
        self.writer(workSheet, 1, 6, df, "kurtosis")
        workBook.close()
        # print(listofKurtosis)
        # print(listofSkew)
        # print(listofAutoCurr)
        # print(listofMax)
        # print(listofMin)
        # print(listofMean)

    def writer(self,worksheet ,x, y, file, data):
        for i in range(len(file[data])):
            worksheet.write(x, y, file[data][i])
            x+=1

    def calculateBidAskAvgReturns(self):
        self.bidAskAvgReturnsOne = []
        self.bidAskAvgReturnsTwo = []
        i = 0
        while i < (len(self.bidAskAvg1)-1):
            self.bidAskAvgReturnsOne.append((np.log(self.bidAskAvg1[i]) - np.log(self.bidAskAvg1[i+1])))
            i+=1
        i=0
        while i < (len(self.bidAskAvg2)-1):
            self.bidAskAvgReturnsTwo.append((np.log(self.bidAskAvg2[i]) - np.log(self.bidAskAvg2[i+1])))
            i+=1

        size = len(self.bidAskAvgReturnsOne)


        self.groupAvg = []
        for i in range(size):
            self.groupAvg.append( (self.bidAskAvgReturnsOne[i] + self.bidAskAvgReturnsTwo[i]) / 2 )



    def fixTimeFormat(self):
        self.time1= []
        self.time2= []
        size = len(self.bidAskAvgReturnsOne)
        for i in range(size):
                self.time1.append(self.dataSet1["Time"][i])

        for i in range(size):
                self.time2.append(self.dataSet2["Time"][i])


        self.floatTimes1 = []
        self.floatTimes2 = []
        for i in range(size):
            hour = int(self.time1[i][0:2])
            minute = int(self.time1[i][3:5])
            second = int(self.time1[i][6:8])
            self.floatTimes1.append(((datetime.timedelta(hours=hour, minutes=minute, seconds=second).total_seconds() )))

        for i in range(size):
            hour = int(self.time2[i][0:2])
            minute = int(self.time2[i][3:5])
            second = int(self.time2[i][6:8])
            self.floatTimes2.append(((datetime.timedelta(hours=hour, minutes=minute, seconds=second).total_seconds() )))


    def startPlottingVolatility(self):
        plt.clf()
        plt.close()
        plt.figure(figsize=(48, 48))
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize = (48,48))
        c = 0
        for i in range(3):
            for j in range(6):
                data = np.vstack([self.df[c]["VolatilityOne"], self.df[c]["VolatilityAvg"]]).T
                minmax = sklearn.preprocessing.MinMaxScaler()
                data = minmax.fit_transform(data)

                mini = c
                maxi = c + 1
                c += 1
                for o in range(len(data)):
                    for t in range(len(data[o])):
                        data[o][t] = (data[o][t] + mini)
                bins = np.arange(mini, maxi, .1)

                axes[i, j].hist(data, bins, alpha=0.5, label=["VolatilityOne", "VolatilityAvg"])
                axes[i, j].legend(loc="upper right")
        plt.title("GrpAskAvgVolatilityOneHisto", fontsize=50, y=3.5, x=-2.5)
        fig.savefig("GrpAskAvgVolatilityOneHisto.png")
        plt.clf()
        plt.close()

        fig, axes = plt.subplots(nrows=3, ncols=6, figsize = (48,48))
        c = 0
        for i in range(3):
            for j in range(6):
                data = np.vstack([self.df2[c]["VolatilityTwo"], self.df2[c]["VolatilityAvg"]]).T
                minmax = sklearn.preprocessing.MinMaxScaler()
                data = minmax.fit_transform(data)
                mini = c
                maxi = c + 1
                # print(data[0][0], "DATA")
                # print(data)
                for o in range(len(data)):
                    for t in range(len(data[o])):
                        data[o][t] = (data[o][t] + mini)
                c += 1
                # print(data)
                bins = np.arange(mini, maxi, .1)
                axes[i, j].hist(data, bins, alpha=0.5, label=["VolatilityTwo", "VolatilityAvg"])
                axes[i, j].legend(loc="upper right")
        plt.title("GrpAskAvgVolatilityTwoHisto", fontsize=50, y=3.5, x=-2.5)
        fig.savefig("GrpAskAvgVolatilityTwoHisto.png")
        plt.clf()
        plt.close()
        plt.figure(figsize=(48, 48))

    def startPlottingBidAskReturn(self):
        # time = self.df[5]["Time"].values
        # askbid = self.df[5]["AskBidAvg"].values
        # grpavg = self.df[5]["GrpAvg"].values
        # minmax = sklearn.preprocessing.MinMaxScaler()
        # time = minmax.fit_transform(time)
        # askbid = minmax.fit_transform(askbid)
        # grpavg = minmax.fit_transform(grpavg)
        # tf = pd.DataFrame()
        # i = len(time-2)
        #
        # tf["Time"] = time
        # tf["AskBidAvg"] = askbid
        # tf["GrpAvg"] = grpavg
        # testdf = pd.melt(tf, id_vars="Time", value_vars=["AskBidAvg", "GrpAvg"])
        # print(testdf)
        #
        # sns.set(style="whitegrid", color_codes=True  )
        # plt.figure(figsize=(24,24))
        # tata = sns.violinplot(x=testdf["Time"], y=testdf["variable"], hue= testdf["variable"], split=True)
        # sns.despine(left=True)
        # tata.figure.savefig("comparison.png")
        # with open("ourdata.txt", "w") as filewrite:
        #     filewrite.write("Time         AskBidAvg             GrpAvg\n")
        #     for i in range(40):
        #         word = str(self.df[5]["Time"][i]) + "    :       "  + str(self.df[5]["AskBidAvg"][i]) + "    :       "  + str(self.df[5]["GrpAvg"][i] )+ " \n"
        #         filewrite.write(word)
        #     filewrite.close()
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize = (48,48))
        c=0
        plt.title("GrpAskAvgReturnsOneHisto", fontsize=50, y=3.5, x=-2.5)
        for i in range(3):
            for j in range(6):
                data = np.vstack([self.df[c]["AskBidAvg"], self.df[c]["GrpAvg"]]).T
                minmax = sklearn.preprocessing.MinMaxScaler()
                data = minmax.fit_transform(data)

                mini = c
                maxi = c+1
                c+=1
                for o in range(len(data)):
                    for t in range(len(data[o])):
                        data[o][t] = (data[o][t] + mini)
                bins = np.arange(mini, maxi, .1)
                axes[i,j].hist(data, bins, alpha=0.5 ,label=["AskBidAvg", "GrpAvg"])
                axes[i,j].legend(loc="upper right")

        fig.savefig("GrpAskAvgReturnsOneHisto.png")
        plt.clf()
        plt.close()


        fig, axes = plt.subplots(nrows=3, ncols=6, figsize = (48,48))
        c=0
        for i in range(3):
            for j in range(6):
                data = np.vstack([self.df2[c]["AskBidAvg"], self.df2[c]["GrpAvg"]]).T
                minmax = sklearn.preprocessing.MinMaxScaler()
                data = minmax.fit_transform(data)
                mini = c
                maxi = c+1
                # print(data[0][0], "DATA")
                # print(data)
                for o in range(len(data)):
                    for t in range(len(data[o])):
                        data[o][t] = (data[o][t] + mini)
                c+=1
                # print(data)
                bins = np.arange(mini, maxi, .1)
                axes[i,j].hist(data,bins,  alpha=0.5 ,label=["AskBidAvg", "GrpAvg"])
                axes[i,j].legend(loc="upper right")
        plt.title("GrpAskAvgReturnsTwoHisto", fontsize=50, y=3.5, x=-2.5)
        fig.savefig("GrpAskAvgReturnsTwoHisto.png")
        plt.clf()
        plt.close()
        plt.figure(figsize=(48, 48))

        # fig, axes = plt.subplots(nrows = 3, ncols = 6, figsize = (24,24))
        # c = 0
        # for i in range(3):
        #     for j in range(6):
        #         axes[i,j].violinplot(self.df[c]["Time"])
        #         c+=1
        # fig.savefig("file1.png")
        #
        # fig, axes = plt.subplots(nrows = 3, ncols = 6, figsize = (24,24))
        # c = 0
        # for i in range(3):
        #     for j in range(6):
        #         axes[i,j].violinplot(self.df2[c]["Time"])
        #         c+=1
        # fig.savefig("file2.png")
        #
        # fig, axes = plt.subplots(nrows = 3, ncols = 6, figsize = (24,24))
        # c = 0
        # for i in range(3):
        #     for j in range(6):
        #         axes[i,j].violinplot(self.df[c]["AskBidAvg"])
        #         c+=1
        #
        # fig.savefig("file1AskBidAVG.png")
        #
        # fig, axes = plt.subplots(nrows = 3, ncols = 6, figsize = (24,24))
        # c = 0
        # for i in range(3):
        #     for j in range(6):
        #         axes[i,j].violinplot(self.df2[c]["AskBidAvg"])
        #         c+=1
        # fig.savefig("file2AskBidAVG.png")
        #
        #
        #
        # fig, axes = plt.subplots(nrows = 3, ncols = 6, figsize = (24,24))
        # c = 0
        # for i in range(3):
        #     for j in range(6):
        #         axes[i,j].violinplot(self.df[c]["GrpAvg"])
        #         c+=1
        # fig.savefig("file1GrpAvg.png")
        #
        #
        # fig, axes = plt.subplots(nrows = 3, ncols = 6, figsize = (24,24))
        # c = 0
        # for i in range(3):
        #     for j in range(6):
        #         axes[i,j].violinplot(self.df2[c]["GrpAvg"])
        #         c+=1
        # fig.savefig("file2GrpAvg.png")

    def stackedPlot(self):
        fnx = lambda :np.random.randint(3, 10, 10)
        y = np.row_stack((fnx(), fnx(), fnx()))
        x = np.arange(18)
        ystack = np.cumsum(y, axis=0)
        # print(type(ystack), "YSTACKCSKS")
        #
        # print(fnx)
        # print("------------")
        # print(y)
        # print("------------")
        # print(x)
        # print("------------")
        # print(ystack, "YSTACK")
        # print(self.corrOneGroup)
        # print(self.corrOneTwo)
        # print(self.corrTwoGroup)
        # print(self.corrOneGroup[0])
        # print(self.corrOneTwo[0])
        # print(self.corrTwoGroup[0])
        # y=[]
        # for i in range(len(self.corrOneGroup)):
        #     y.append([self.corrOneTwo[i][1], self.corrOneGroup[i][1], self.corrTwoGroup[i][1]])
        y1 = []
        for i in range(len(self.corrOneTwo)):
            y1.append(self.corrOneTwo[i][1])
        y2 = []
        for i in range(len(self.corrOneTwo)):
            y2.append(self.corrOneTwo[i][1] *0.4)
        y3 = []
        for i in range(len(self.corrOneGroup)):
            y3.append(self.corrTwoGroup[i][1])
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)

        # y = np.row_stack((self.corrOneGroup(), self.corrTwoGroup(), self.corrOneTwo()))
        # print(ystack)
        # print(type(ystack))

        plt.clf()
        plt.close()
        plt.figure(figsize=(48, 48))
        # ys= np.row_stack(x, yarray)
        x = np.arange(18)
        # print(ys)
        # print(len(ys))
        # print(x)
        # print(len(x))
        # fig = plt.figure()
        fig, ax1 = plt.stackplot(x, y1,y2)

        plt.title("StackedPlot CorrOneTwo, *0.4", fontsize=50)
        fig.figure.savefig("StackedPlot.png")


        # ax1.fill_between(x, 0, ystack[0,:], facecolor="#CC6666", alpha=.7)
        # ax1.fill_between(x, ystack[0,:], ystack[1,:], facecolor="#1DACD6", alpha=.7)
        # ax1.fill_between(x, ystack[1,:], ystack[2,:], facecolor="#6E5160")
    def calculateCorrGroups(self):
        corrOneTwo = []
        corrOneGroup = []
        corrTwoGroup = []

        x = []
        y = []
        for i in range(len(self.df)):
            if((len(self.df[i]["AskBidAvg"]) < len(self.df2[i]["AskBidAvg"]))):
                minimum = len(self.df[i]["AskBidAvg"])
            else:
                minimum = len(self.df2[i]["AskBidAvg"])
            for j in range(minimum):
                x.append(self.df[i]["AskBidAvg"][j])
                y.append(self.df2[i]["AskBidAvg"][j])
            value = pearsonr(x, y)
            corrOneTwo.append([value[0], value[1]])
            x= []
            y= []

        x = []
        y = []
        for i in range(len(self.df)):
            if((len(self.df[i]["AskBidAvg"]) < len(self.df[i]["GrpAvg"]))):
                minimum = len(self.df[i]["AskBidAvg"])
            else:
                minimum = len(self.df[i]["GrpAvg"])
            for j in range(minimum):
                x.append(self.df[i]["AskBidAvg"][j])
                y.append(self.df[i]["GrpAvg"][j])
            value = pearsonr(x, y)
            corrOneGroup.append([value[0], value[1]])
            x= []
            y= []
        x = []
        y = []
        for i in range(len(self.df)):
            if((len(self.df2[i]["AskBidAvg"]) < len(self.df2[i]["GrpAvg"]))):
                minimum = len(self.df2[i]["AskBidAvg"])
            else:
                minimum = len(self.df2[i]["GrpAvg"])
            for j in range(minimum):
                x.append(self.df2[i]["AskBidAvg"][j])
                y.append(self.df2[i]["GrpAvg"][j])
            value = pearsonr(x, y)
            corrTwoGroup.append([value[0], value[1]])
            x= []
            y= []
        self.corrOneTwo = corrOneTwo
        self.corrOneGroup = corrOneGroup
        self.corrTwoGroup = corrTwoGroup
        test = pd.DataFrame()
        fullcorr = []
        for i in range(len(self.corrOneTwo)):
            fullcorr.append(self.corrOneTwo[i])
        tta= []
        for i in range(len(corrOneTwo)):
            tta.append([corrOneTwo[i][0], corrOneTwo[i][1]])
        test["a"] = tta
        # test["b"] = corrOneGroup
        # test["c"] = corrTwoGroup
        # for i in range(len(self.corrOneGroup)):
        #     a.append([self.corrOneGroup[i], self.corrTwoGroup[i], self.corrOneTwo[i]])
        total = np.vstack([self.corrOneGroup, self.corrTwoGroup, self.corrOneTwo]).T
        total2 = np.vstack([self.corrOneGroup, self.corrTwoGroup, self.corrOneTwo])
        # print(total)
        # y = []
        # for i in range(len(self.corrOneGroup[2])):
        #     if(i == 0):
        #         y.append(0)
        #     else:
        #         y.append(0 + (1 / i))
        # test["d"] = y


        # testdf = pd.melt(tf, id_vars="Time", value_vars=["AskBidAvg", "GrpAvg"])
        # print(testdf)
        # minmax = sklearn.preprocessing.MinMaxScaler()
        # data = minmax.fit_transform(total)
        # result = pd.melt(test, id_vars="d", value_vars=["a", "b", "c"])
        # for i in range(len(result["d"])):
        #     print(result["d"][i])
        # test = sns.heatmap(total, annot=True, cmap="plasma")
        # test.figure.savefig("tasta.png")
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(24, 24))
        # axes.imshow(total, cmap="hot", interpolation="nearest")
        # fig.savefig("tastat.png")

        # fig, ax = plt.subplots()
        # fig.set_size_inches(48,48)
        # ax = sns.heatmap(total,fmt=".0%" , linewidths=.1)
        # fig.figure.savefig("test.png")
        fullCorrelation = []
        for i in range(len(corrOneTwo)):
            fullCorrelation.append([ corrOneTwo[i][1], corrOneGroup[i][1], corrTwoGroup[i][1]])



        savetoCSV = []
        for i in range(len(corrOneTwo)):
            savetoCSV.append([corrOneTwo[i][0], corrOneGroup[i][0], corrTwoGroup[i][0] ])

        with open("corrOneTwo_corrOneGroup_corrTwoGroup.txt", "w") as writer:
            writer.write("corrOneTwo    corrOneGroup    corrTwoGroup\n")
            for i in range(len(savetoCSV)):
                sentence = str(savetoCSV[i][0])+ "\t"+ str(savetoCSV[i][1])+ "\t"+ str(savetoCSV[i][2])+ "\n"
                writer.write(sentence)


        print("Plotting Heatmaps of CorrOneTwo, CorrOneGroup And CorrTwotGroup")

        plt.figure(figsize=(48,48))
        cr1 = sns.cubehelix_palette( as_cmap= True, rot = 0, light=1)
        fig1 = sns.heatmap(fullCorrelation,fmt="d" , linewidths=1, cmap=cr1)
        plt.xlabel("corrOneTwo                          -> corrOneGroup                              -> corrTwoGroup", fontsize = 50)
        plt.title("Heatmap Between CorrOneTwo, CorrOneGroup and CorrTwoGroup", fontsize=50)
        plt.xticks(fontsize=50)
        plt.yticks([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fontsize=50)
        plt.tight_layout()
        fig1.figure.savefig("fullcorr.png")
        plt.clf()
        plt.close()
        plt.figure(figsize=(48,48))
        cr1 = sns.cubehelix_palette(as_cmap= True, rot = 0, light=1)
        fig1 = sns.heatmap(total,fmt="d" , linewidths=1, cmap=cr1)
        plt.xticks(fontsize=50)
        plt.yticks([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fontsize=50)
        plt.tight_layout()
        fig1.figure.savefig("total.png")
        plt.clf()
        plt.close()
        plt.figure(figsize=(48, 48))
        cr2 = sns.cubehelix_palette(as_cmap= True, rot = -.2, light=1, dark=.1)
        fig2 = sns.heatmap(self.corrOneTwo,fmt="d" , linewidths=1, cmap=cr2)
        plt.xticks(fontsize=50)
        plt.yticks([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fontsize=50)
        plt.tight_layout()
        plt.title("CorrOneTwo", fontsize=50)
        fig2.figure.savefig("corrOneTwo.png")
        plt.clf()
        plt.close()
        plt.figure(figsize=(48, 48))
        cr2 = sns.cubehelix_palette(as_cmap= True, rot = 0, light=1)
        fig2 = sns.heatmap(total2,fmt="d" , linewidths=1, cmap=cr2)
        plt.xticks(fontsize=50)
        plt.yticks([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fontsize=50)
        plt.tight_layout()
        fig2.figure.savefig("total2.png")
        plt.clf()
        plt.close()
        plt.figure(figsize=(48, 48))
        cr3 = sns.cubehelix_palette(as_cmap= True, rot = -.3, light=1)
        fig3 = sns.heatmap(self.corrOneGroup,fmt="d" , linewidths=1, cmap=cr3)
        plt.xticks(fontsize=50)
        plt.yticks([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fontsize=50)
        plt.tight_layout()
        plt.title("corrOneGroup", fontsize=50)
        fig3.figure.savefig("corrOneGroup.png")
        plt.clf()
        plt.close()
        plt.figure(figsize=(48, 48))
        cr4 = sns.cubehelix_palette(as_cmap= True, rot = -.3, light=1)
        fig4 = sns.heatmap(self.corrTwoGroup,fmt="d" , linewidths=1, cmap=cr4)
        plt.xticks(fontsize=50)
        plt.yticks([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fontsize=50)
        plt.tight_layout()
        plt.title("corrTwoGroup", fontsize=50)
        fig4.figure.savefig("corrTwoGroup.png")


        sharpe1 = []
        sharpe2 = []
        sharpe3 = []
        for i in range(len(self.df)):
            sharpe1.append( (np.sqrt(60) * ( (np.mean(self.df[i]["AskBidAvg"])) / (np.std(self.df[i]["AskBidAvg"])) )) )
        for i in range(len(self.df2)):
            sharpe2.append((np.sqrt(60) * ((np.mean(self.df2[i]["AskBidAvg"])) / (np.std(self.df2[i]["AskBidAvg"])))))
        for i in range(len(self.df)):
            sharpe3.append((np.sqrt(60) * ((np.mean(self.df[i]["GrpAvg"])) / (np.std(self.df[i]["GrpAvg"])))))

        stackedSharpe = np.vstack([sharpe1, sharpe2, sharpe3]).T
        with open("sharpe.txt", "w") as writer:
            writer.write("Sharp Values")
            for i in range(len(stackedSharpe)):
                sentence = str(stackedSharpe[i][0]) + "\t\t" + str(stackedSharpe[i][1]) + "\t\t" + str(stackedSharpe[i][2]) + "\n"
                writer.write(sentence)
        plt.clf()
        plt.close()
        print("Plotting Sharpe Heatmap...")
        plt.figure(figsize=(48, 48))
        cr5 = sns.cubehelix_palette(as_cmap= True, rot = -.3, light=1)
        fig5 = sns.heatmap(stackedSharpe, vmin=-1, vmax=1, center=0.0,fmt="d" , linewidths=1, cmap=cr5)
        plt.title("Sharpe HeatMap", fontsize=50)
        plt.xlabel("corrOneTwo                          -> corrOneGroup                              -> corrTwoGroup",fontsize=50)
        plt.xticks(fontsize=50)
        plt.yticks([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fontsize=50)
        plt.tight_layout()
        fig5.figure.savefig("SharpeHeatMap.png")
        plt.clf()
        plt.close()

        maxValue = -5
        minValue = 2222
        for i in range(len(sharpe1)):
            if(minValue > sharpe1[i]):
                minValue = sharpe1[i]
            if(minValue > sharpe2[i]):
                minValue = sharpe2[i]
            if(minValue > sharpe3[i]):
                minValue = sharpe3[i]
        for i in range(len(sharpe1)):
            if(maxValue < sharpe1[i]):
                maxValue = sharpe1[i]
            if(maxValue < sharpe2[i]):
                maxValue = sharpe2[i]
            if(maxValue < sharpe3[i]):
                maxValue = sharpe3[i]

        plt.figure(figsize=(48, 48))
        cr5 = sns.cubehelix_palette(as_cmap= True, rot = -.3, light=1)
        fig5 = sns.heatmap(stackedSharpe,fmt="d" , linewidths=1, cmap=cr5, center=0.0, vmin= minValue, vmax= maxValue)
        plt.title("Sharpe HeatMap", fontsize=50)
        plt.xlabel("corrOneTwo                          -> corrOneGroup                              -> corrTwoGroup",fontsize=50)
        plt.xticks(fontsize=50)
        plt.yticks([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fontsize=50)
        red_Patch = mpatches.Patch(color="red", label="The Red Data")
        plt.legend(handles= [red_Patch])

        plt.tight_layout()
        fig5.figure.savefig("SharpeHeatMap2.png")
        plt.clf()
        plt.close()
        plt.figure(figsize=(48, 48))
        cr5 = sns.cubehelix_palette(as_cmap= True, rot = -.3, light=1)
        fig5 = sns.heatmap(stackedSharpe,fmt="d" , linewidths=1, cmap=cr5)
        plt.title("Sharpe HeatMap", fontsize=50)
        plt.xlabel("corrOneTwo                          -> corrOneGroup                              -> corrTwoGroup",fontsize=50)
        plt.xticks(fontsize=50)
        plt.yticks([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fontsize=50)
        word = "Minumum : " + str(minValue) + "\nMaximum : "+ str(maxValue)
        red_Patch = mpatches.Patch( label=word)
        plt.legend(handles= [red_Patch], fontsize= 50)

        plt.tight_layout()
        fig5.figure.savefig("SharpeHeatMap3.png")
        plt.clf()
        plt.close()
        # sns.set()
        # sns.heatmap([corrOneGroup[2], corrTwoGroup[2]])
        # plt.show()
        # print("sassa")
        # # self.CorrOneTwo = pearsonr(self.bidAskAvgReturnsOne, self.bidAskAvgReturnsTwo)
        # # self.CorrOneGroup = pearsonr(self.bidAskAvgReturnsOne, self.groupAvg)
        # # self.CorrTwoGroup = pearsonr(self.bidAskAvgReturnsTwo, self.groupAvg)
        #
        #
        # # print("self.CorrOneTwo: ", self.CorrOneTwo)
        # # print("self.CorrOneGroup: ", self.CorrOneGroup)
        # # print("self.CorrTwoGroup: ", self.CorrTwoGroup)
        # sad
        # sharpe = (np.sqrt(60) * (np.mean(self.bidAskAvgReturnsOne))) / np.std(self.bidAskAvgReturnsOne)
        # sns.set(context="paper", font = "monospace")
        # f, ax = plt.subplots(figsize=(12, 9))
        # # data = []
        # # for i in range(len(self.bidAskAvgReturnsOne)):
        # #     data.append([self.bidAskAvgReturnsOne[i], self.bidAskAvgReturnsTwo[i],self.groupAvg[i]])
        #
        # df = pd.DataFrame({"Time":self.listofFixedTimes1, "BidAskAvgone":self.bidAskAvgReturnsOne, "BidAskAvgTwo":self.bidAskAvgReturnsTwo, "GroupAvg":self.groupAvg})
        # # result = df.pivot(index="Time", columns=("Time"), values=["BidAskAvgone","BidAskAvgTwo","GroupAvg"])
        # # sns.heatmap(vmin= -5, vmax=5, center=0.0, yticklabels=self.listofFixedTimes1,xticklabels= [self.bidAskAvgReturnsOne, self.bidAskAvgReturnsTwo, self.groupAvg], data=[self.bidAskAvgReturnsOne, self.bidAskAvgReturnsTwo, self.groupAvg])
        # sns.heatmap(vmin= -5, vmax=5, center=0.0,data=df)
        #
        # f.tight_layout()
        # plt.show()

    def calculateVolatility(self):
        self.volatilityOne = []
        self.volatilityTwo = []
        self.groupVolatility = []
        for i in range(len(self.bidAskAvgReturnsOne)):
            self.volatilityOne.append(np.sqrt(abs(self.bidAskAvgReturnsOne[i])))

        for i in range(len(self.bidAskAvgReturnsTwo)):
            self.volatilityTwo.append(np.sqrt(abs(self.bidAskAvgReturnsTwo[i])))

        for i in range(len(self.bidAskAvgReturnsTwo)):
            self.groupVolatility.append(((self.volatilityOne[i] + self.volatilityTwo[i])/2))



    def splitMinute(self):
        checker = 1.0
        print(len(self.df))
        for i in range(len(self.df)):
            for j in range(len(self.df[i]["Time"].values)):
                if(self.df[i]["Time"][j] / 60 >= checker):
                    checker+=1
                    print(self.df[i]["Time"][j])
        print(checker)
        print(self.df)

    def splitDataFrames(self):
        self.df = [pd.DataFrame() for i in range(18)]
        self.df2 = [pd.DataFrame() for i in range(18)]
        size = len(self.floatTimes1)

        time = []
        grpavg =[]
        bidaskavg = []
        volatilityOne = []
        volatilityAvg = []
        checker = 1
        dataFrameCounter = 0
        for i in range(size-1):
            if(i+1 == size-1):
                self.df[dataFrameCounter]["Time"] = time
                self.df[dataFrameCounter]["AskBidAvg"] = bidaskavg
                self.df[dataFrameCounter]["GrpAvg"] = grpavg
                self.df[dataFrameCounter]["VolatilityOne"] = volatilityOne
                self.df[dataFrameCounter]["VolatilityAvg"] = volatilityAvg
                break
            if(self.listofFixedTimes1[i] == checker):
                checker+=1
                self.df[dataFrameCounter]["Time"] = time
                self.df[dataFrameCounter]["AskBidAvg"] = bidaskavg
                self.df[dataFrameCounter]["GrpAvg"] = grpavg
                self.df[dataFrameCounter]["VolatilityOne"] = volatilityOne
                self.df[dataFrameCounter]["VolatilityAvg"] = volatilityAvg
                grpavg =[]
                bidaskavg =[]
                time = []
                volatilityAvg = []
                volatilityOne = []
                dataFrameCounter+=1
                if(dataFrameCounter == 18):
                    break
            volatilityAvg.append(self.groupVolatility[i])
            time.append(self.floatTimes1[i])
            grpavg.append(self.groupAvg[i])
            volatilityOne.append(self.volatilityOne[i])
            bidaskavg.append(self.bidAskAvgReturnsOne[i])

        # with open("data.txt", "w") as filewriter:
        #     for i in range(len(self.df[4])):
        #         string = str(self.df[4]["Time"][i]) + "     :    " + str(self.df[4]["AskBidAvg"][i])+ "     :   " + str(self.df[4]["GrpAvg"][i]) + "\n"
        #         filewriter.write(string)
        #
        # filewriter.close()

        time = []
        grpavg = []
        bidaskavg = []
        volatilityTwo = []
        volatilityAvg = []
        checker = 1
        dataFrameCounter = 0
        for i in range(size - 1):
            if (i + 1 == size - 1):
                self.df2[dataFrameCounter]["Time"] = time
                self.df2[dataFrameCounter]["AskBidAvg"] = bidaskavg
                self.df2[dataFrameCounter]["GrpAvg"] = grpavg
                self.df2[dataFrameCounter]["VolatilityTwo"] = volatilityTwo
                self.df2[dataFrameCounter]["VolatilityAvg"] = volatilityAvg
                break
            if (self.listofFixedTimes2[i] == checker):
                checker += 1
                self.df2[dataFrameCounter]["Time"] = time
                self.df2[dataFrameCounter]["AskBidAvg"] = bidaskavg
                self.df2[dataFrameCounter]["GrpAvg"] = grpavg
                self.df2[dataFrameCounter]["VolatilityTwo"] = volatilityTwo
                self.df2[dataFrameCounter]["VolatilityAvg"] = volatilityAvg
                grpavg = []
                bidaskavg = []
                time = []
                volatilityAvg = []
                volatilityTwo = []
                dataFrameCounter += 1
                if(dataFrameCounter == 18):
                    break
            time.append(self.floatTimes2[i])
            grpavg.append(self.groupAvg[i])
            volatilityAvg.append(self.groupVolatility[i])
            volatilityTwo.append(self.volatilityTwo[i])
            bidaskavg.append(self.bidAskAvgReturnsTwo[i])





print("Reading Data from csv...")
visualizeData = dataVisualizer("data1.csv", "data2.csv")
print("Calculating Bid Ask Avg...")
visualizeData.calculateBidAskAvg()
print("Calculating Bid Ask Avg Returns...")
visualizeData.calculateBidAskAvgReturns()
print("Generating Simple Tabular Data...")
visualizeData.generateTaularData()
print("Turning Dates to Floats...")
visualizeData.fixTimeFormat()
print("Normalizing Dates...")
visualizeData.splitHours()
print("Calculating Volatility...")
visualizeData.calculateVolatility()
print("Splitting to 1 hr Format...")
visualizeData.splitDataFrames()
print("Plotting Histograms between BidAskAvgReturns One and GroupAvgReturns, BidAskAvgReturnsTwo and GroupAvgReturns...")
visualizeData.startPlottingBidAskReturn()
print("Plotting Histograms between BidAskAvgVilatilityOne and GroupAvgVolatility, BidAskAvgVolatilityTwo and GroupAvgVolatility...")
visualizeData.startPlottingVolatility()
print("Calculating correlation Arrays")
visualizeData.calculateCorrGroups()
print("Plotting stacked line plot of CorrOneTwo, CorrOneGroup, CorrTwoGroup...")
visualizeData.stackedPlot()
print("Writing pdf File...")
visualizeData.writePDF()
print("Done.")
# print("Splitting Minutes....")
# visualizeData.splitMinute()


# a = [5,1,1]
# b= [8,4,25,235,235]
# c = [124,124,15]
# plt.violinplot([a,b,c])
# plt.show()
# test = pd.DataFrame()
# s = []
# for i in range(10000):
#     s.append(random.randrange(19))
#
# test["Hour"] = s
#
# s= []
# for i in range(10000):
#     s.append(random.randrange(-1,1))
#
# test["average"] = s
#
# s = []
# for i in range(10000):
#     s.append(random.randrange(200))
#
# fs = 10  # fontsize
# pos = [1, 2, 4, 5, 7, 8]
# data = [np.random.normal(0, std, size=100) for std in pos]
#
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
#
# axes[0, 0].violinplot(data, pos, points=20, widths=0.3,
#                       showmeans=True, showextrema=True, showmedians=True)
# axes[0, 0].set_title('Custom violinplot 1', fontsize=fs)
#
# axes[0, 1].violinplot(data, pos, points=40, widths=0.5,
#                       showmeans=True, showextrema=True, showmedians=True,
#                       bw_method='silverman')
# axes[0, 1].set_title('Custom violinplot 2', fontsize=fs)
#
# axes[0, 2].violinplot(data, pos, points=60, widths=0.7, showmeans=True,
#                       showextrema=True, showmedians=True, bw_method=0.5)
# axes[0, 2].set_title('Custom violinplot 3', fontsize=fs)
#
# axes[1, 0].violinplot(data, pos, points=80, vert=False, widths=0.7,
#                       showmeans=True, showextrema=True, showmedians=True)
# axes[1, 0].set_title('Custom violinplot 4', fontsize=fs)
#
# axes[1, 1].violinplot(data, pos, points=100, vert=False, widths=0.9,
#                       showmeans=True, showextrema=True, showmedians=True,
#                       bw_method='silverman')
# axes[1, 1].set_title('Custom violinplot 5', fontsize=fs)
#
# axes[1, 2].violinplot(data, pos, points=200, vert=False, widths=1.1,
#                       showmeans=True, showextrema=True, showmedians=True,
#                       bw_method=0.5)
# axes[1, 2].set_title('Custom violinplot 6', fontsize=fs)
#
# for ax in axes.flatten():
#     ax.set_yticklabels([])
#
# fig.suptitle("Violin Plotting Examples")
# fig.subplots_adjust(hspace=0.4)
# plt.show()
# a = pd.DataFrame()
# c = []
# z = []
# x = []
# st = ["a", "b"]
# te = []
# for j in range(1000):
#     c.append(random.randrange(0, 10000))
#     z.append(random.randrange(0, 10000))
#     x.append(random.randrange(0, 5000))
#     te.append(st[random.randrange(0,2)])
# # te.pop(2)
# # te.append("sasa")
# a["tata"] = c
# a["papa"] = z
# a["yaya"] = x
# a["baba"] = te
# # test = pd.DataFrame()
# # sns.set(style="white", color_codes=True)
# # f, ax = plt.subplots(figsize = (8,8))
# # sns.violinplot(x="yaya",y= "baba",hue="baba" ,inner ="quartiles" , linewidth= 1,split=True , data=a )
# # plt.show()
#
# bins = []
# data = np.vstack([a["tata"], a["papa"]]).T
# mini = 10000
# maxo=0
# for i in range(len(data)):
#     for j in range(len(data[i])):
#         if mini > data[i][j]:
#             mini = data[i][j]
# for i in range(len(data)):
#     for j in range(len(data[i])):
#         if maxo < data[i][j]:
#             maxo = data[i][j]
#
# bins.append(mini)
# bins.append(maxo)
# print(len(data))
# print(maxo, "maxo")
# print(mini, "mini")
# test = np.arange(mini, maxo, 300)
# plt.hist(data, test ,histtype="bar", rwidth=0.7)
# plt.show()
# plt.hist(data, alpha = 0.7, label= ["tata", "papa"])
# plt.legend(loc= "upper right")
# plt.show()
# ax = sns.violinplot(x= "tata", hue= "papa" , data = a, split= True)
# ax.figure.savefig("yaya.png")# test.figure.savefig("tata.png")
# sns.violinplot(x="yaya",y= "baba",hue="baba" ,inner ="quartiles", palette={"a":"b","b":"y"} , linewidth= 1,split=True , data=a )

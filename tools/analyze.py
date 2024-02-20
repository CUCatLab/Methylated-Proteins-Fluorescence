import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame as df
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import ipywidgets as ipw
from ipywidgets import Button, Layout
from lmfit import model, Model
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel
import re
import os
from os import listdir
from os.path import isfile, join, dirname
import sys
from pathlib import Path
from IPython.display import clear_output
from pylab import rc

pio.renderers.default = 'notebook+plotly_mimetype'
pio.templates.default = 'simple_white'
pio.templates[pio.templates.default].layout.update(dict(
    title_y = 0.95,
    title_x = 0.5,
    title_xanchor = 'center',
    title_yanchor = 'top',
    legend_x = 0,
    legend_y = 1,
    legend_traceorder = "normal",
    legend_bgcolor='rgba(0,0,0,0)',
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=50, #top margin
        )
))


class dataTools () :

    def __init__(self,parFile,folder='') :
        
        if folder == '' :
            with open(parFile+'.yaml', 'r') as stream:
                self.par = yaml.safe_load(stream)
        
        self.parFile = parFile
    
    def fileList (self) :

        with open(self.parFile+'.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        if 'dataFolder' in par :
            par.pop('dataFolder')

        filenames = list()
        for filename in par :
            filenames.append(filename)
        
        return filenames
    
    def LoadData(self,folder) :

        files = list()
        columns = list()
        columns.append('nm')
        for file in os.listdir(folder):
            if re.search('.txt', file):
                files.append(file)
                columns.append(file.replace('.txt',''))

        for idx,file in enumerate(files):
            tempData = pd.read_csv(folder+'/'+file,sep='\t',header=None,skiprows=14)
            if idx == 0:
                data = tempData
            else :
                tempDataColumn = tempData[1]
                data = pd.concat([data,tempDataColumn], axis=1)
        data = data.reindex(sorted(data.columns), axis=1)

        data.columns = columns
        data = data.set_index('nm')

        return data

    def trimData(self,data,range) :
        
        Min = np.min(range)
        Max = np.max(range)
        Mask = np.all([data.index>Min,data.index<Max],axis=0)
        data = data[Mask]
        
        return data

    def PlotRegion(self,region,normalize=True) :

        with open(self.parFile+'.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        fig = go.Figure()
        fig.update_layout(xaxis_title="Energy (eV)",yaxis_title="Intensity (au)",title=region,font=dict(size=18),
            autosize=False,width=1000,height=600)
        filenames = list()
        for filename in par :
            filenames.append(filename)

        data = {}
        for filename in filenames :
            data[filename] = {}
            newdata = self.LoadData(filename,normalize)
            if region in newdata :
                data[filename][region] = {}
                data[filename][region]['x'] = newdata[region]['x']
                data[filename][region]['y'] = newdata[region]['y']
                name = filename
                if 'Temperature' in par[filename] :
                    name += ', '+str(par[filename]['Temperature'])+' K'
                fig.add_trace(go.Scatter(x=data[filename][region]['x'], y=data[filename][region]['y'], name=name))

        self.data = data
        fig.show()

    def PlotFile(self,filename,region,normalize=True) :

        data = self.LoadData(filename,normalize)

        fig = go.Figure()
        fig.update_layout(xaxis_title="Energy (eV)",yaxis_title="Intensity (au)",title=filename,font=dict(size=18),
            autosize=False,width=1000,height=600)
        for col in data[region] :
            if 'y' in col :
                fig.add_trace(go.Scatter(x=data[region]['x'], y=data[region][col], name=col))

        self.data = data
        fig.show()


class fitTools :
    
    def __init__(self) :

        pass

    def LoadPar(self,parFile,folder='') :
        if folder == '' :
            with open(parFile+'.yaml', 'r') as stream:
                self.par = yaml.safe_load(stream)

        self.dt = dataTools(parFile)
        self.parFile = parFile
    
    def SetModel(self, Data, par) :
        
        ModelString = list()
        for Peak in par :
            ModelString.append((Peak,par[Peak]['model']))
        
        for Model in ModelString :
            try :
                FitModel
            except :
                if Model[1] == 'Constant' :
                    FitModel = ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = VoigtModel(prefix=Model[0]+'_')
                if Model[1] == 'Shirley' :
                    FitModel = ShirleyBG(prefix=Model[0]+'_')
            else :
                if Model[1] == 'Constant' :
                    FitModel = FitModel + ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = FitModel + LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = FitModel + GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = FitModel + SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = FitModel + VoigtModel(prefix=Model[0]+'_')
        
        ModelParameters = FitModel.make_params()
        names = list()
        for col in Data :
            if 'y' in col :
                names.append(col)
        FitsParameters = df(index=ModelParameters.keys(),columns=names)
        
        self.FitModel = FitModel
        self.ModelParameters = ModelParameters
        self.FitsParameters = FitsParameters
    
    def SetParameters(self, par, Value=None) :
        
        ModelParameters = self.ModelParameters
        
        ParameterList = ['intercept','offset','amplitude','center','sigma']
        Parameters = {'Standard': par}

        for Dictionary in Parameters :
            for Peak in Parameters[Dictionary] :
                for Parameter in Parameters[Dictionary][Peak] :
                    if Parameter in ParameterList :
                        for Key in Parameters[Dictionary][Peak][Parameter] :
                            if Key != 'set' :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+'='+str(Parameters[Dictionary][Peak][Parameter][Key]))
                            else :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+str(Parameters[Dictionary][Peak][Parameter][Key]))
                                    
        self.ModelParameters = ModelParameters
    
    def Fit(self,data,par,**kwargs) :

        self.SetModel(data,par)
        ModelParameters = self.ModelParameters
        
        FitModel = self.FitModel
        
        Fits = df(index=data.index,columns=data.columns)
        FitsParameters = df(index=ModelParameters.keys(),columns=data.columns)
        FitsResults = list()
        FitsComponents = list()
        NInt = list()
        
        for idx,file in enumerate(data.columns) :
            self.SetParameters(par)
            x = data.index
            y = data[file]
            FitResults = FitModel.fit(y, ModelParameters, x=x, nan_policy='omit')
            fit_comps = FitResults.eval_components(FitResults.params, x=x)
            fit_y = FitResults.eval(x=x)
            ParameterNames = [i for i in FitResults.params.keys()]
            for Parameter in (ParameterNames) :
                FitsParameters.loc[Parameter,file] = FitResults.params[Parameter].value
            NInt.append(np.trapz(y,x=x))
            Fits[data.columns[idx]] = fit_y
            FitsResults.append(FitResults)
            FitsComponents.append(fit_comps)
            
            sys.stdout.write(("\rFitting %i out of "+str(len(data.columns))) % (idx+1))
            sys.stdout.flush()

        self.Fits = Fits
        self.FitsParameters = FitsParameters
        self.FitsResults = FitsResults
        self.FitsComponents = FitsComponents
        self.NInt = NInt
    
    def FitData(self) :
        
        with open(self.parFile+'.yaml', 'r') as stream:
            self.par = yaml.safe_load(stream)
        
        data = self.data
        par = self.par

        ##### Trim Data #####
        
        if 'range' in par[self.dataSet.value] :
            data = self.dt.trimData(data, par[self.dataSet.value]['range'])
        
        ##### Fit Data #####
        
        if 'Models' in par[self.dataSet.value] :
            self.Fit(data,par[self.dataSet.value]['Models'])
            Fits = self.Fits
            FitsParameters = self.FitsParameters

        print('\n'+100*'_')
        
        for idx,file in enumerate(data.columns) :
            plt.figure(figsize = [8,3])
            plt.plot(data.index, data[file],'k.', label='Data')
            plt.plot(Fits.index, Fits[file], 'r-', label='Fit')
            plt.xlabel('Energy (eV)'), plt.ylabel('Intensity (au)')
            plt.title(file)
            for Component in self.FitsComponents[idx-1] :
                Peak = Component[:-1]
                if not isinstance(self.FitsComponents[idx-1][Component],float) :
                    if 'assignment' in self.par[self.dataSet.value]['Models'][Peak] :
                        label = self.par[self.dataSet.value]['Models'][Peak]['assignment']
                    else :
                        label = Peak
                    plt.fill(Fits.index, self.FitsComponents[idx][Component], '--', label=label, alpha=0.5)
            plt.show()
            
            Peaks = list()
            for Parameter in FitsParameters.index :
                Name = Parameter.split('_')[0]
                if Name not in Peaks :
                    Peaks.append(Name)
            string = ''
            for Peak in Peaks :
                if 'assignment' in par[self.dataSet.value]['Models'][Peak] :
                    string += par[self.dataSet.value]['Models'][Peak]['assignment'] + ' | '
                else :
                    string += Peak + ' | '
                for Parameter in FitsParameters.index :
                    if Peak == Parameter.split('_')[0] : 
                        string += Parameter.split('_')[1] + ': ' + str(round(FitsParameters[file][Parameter],2))
                        string += ', '
                string = string[:-2] + '\n'
            print(string)
            print(100*'_')
        FitsParameters = FitsParameters.T
        FitsParameters = FitsParameters[np.concatenate((FitsParameters.columns.values[1:],FitsParameters.columns.values[0:1]))]
    
    def UI(self) :

        out = ipw.Output()
    
        def parList(filter='yaml') :
        
            parList = [f for f in os.listdir()]
            for i in range(len(filter)):
                parList = [k for k in parList if filter[i] in k]
            for i in range(len(parList)):
                parList[i] = parList[i].replace('.yaml','')
            
            return parList

        def selecting(value) :
            if value['new'] :
                self.LoadPar(selectParameters.value)
                self.dataSet.options = self.dt.fileList

        selectParameters = ipw.Dropdown(
            options=parList(),
            description='Select Parameters File',
            layout=Layout(width='70%'),
            style = {'description_width': '150px'},
            disabled=False,
        )
        selectParameters.observe(selecting, names='value')

        self.LoadPar(selectParameters.value)

        self.dataSet = ipw.Dropdown(
            options=self.dt.fileList(),
            description='Select Data Set',
            layout=Layout(width='70%'),
            style = {'description_width': '150px'},
            disabled=False,
        )

        def saveData_Clicked(b) :
            with out :
                with pd.ExcelWriter(self.dataSet.value+'.xlsx') as writer :
                    self.data.to_excel(writer, sheet_name="Data")
                    self.Fits.to_excel(writer, sheet_name="Fits")
                    Parameters = self.FitsParameters.transpose()
                    peaks_amplitude = np.zeros((len(self.NInt)))
                    for column in Parameters.columns :
                        if 'amplitude' in column :
                            peaks_amplitude += Parameters[column]
                    Parameters['peaks_amplitude'] = peaks_amplitude
                    Parameters['NInt'] = self.NInt
                    Parameters.to_excel(writer, sheet_name="Parameters")
        saveData = ipw.Button(description="Save Data")
        saveData.on_click(saveData_Clicked)
        
        def loadData_Clicked(b) :
            with out :
                clear_output(True)
                self.data = self.dt.LoadData(self.par[self.dataSet.value]['folder'])
                self.FitData()
                display(saveData)
        loadData = ipw.Button(description="Analyze Data")
        loadData.on_click(loadData_Clicked)
        
        display(selectParameters)
        display(self.dataSet)
        display(loadData)
        display(out)
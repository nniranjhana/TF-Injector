#!/usr/bin/python

import yaml

def config(confFile):
    try:
        fiConfs = open(confFile, "r")
    except IOError:
        print("Unable to open the config file ", confFile)
        return fiConf
    if confFile.endswith(".yaml"):
        fiConf = yaml.load(fiConfs)
    else:
        print("Unsupported file format: ", confFile)
    return fiConf

def mconfig(confFile = None):
    fiConf = {}
    if confFile == None:
        fiConf["Artifact"] = 0
        fiConf["Type"] = "shuffle"
        return fiConf
    fiConf = config(confFile)
    return fiConf

def dconfig(confFile = None):
    fiConf = {}
    if confFile == None:
        fiConf["Type"] = "shuffle"
        return fiConf
    fiConf = config(confFile)
    return fiConf
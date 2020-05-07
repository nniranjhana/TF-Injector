#!/usr/bin/python

import yaml

def config(confFile = None):
    fiConf = {}
    if confFile == None:
        fiConf["Artifact"] = 0
        fiConf["Type"] = "shuffle"
        return fiConf
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
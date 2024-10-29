import numpy as np
import csv
import unittest
from ..tasks import DataFactory as DF

class TestDataFactory(unittest.TestCase, DF.DataFactory):
    def setUp(self):
        # 创建一个临时的 CSV 文件用于测试
        self.test_csv_data = r"E:\算法\项目管理\FitFormAI\仰卧起坐-侧面视角-单侧发力起坐(1).csv"
        self.dataFactory = DF.DataFactory(self.test_csv_data)

    def test_setDataWave(self):
        waveData = self.dataFactory.getWaveData()
        print(waveData['dataNameList'])
        

if __name__ == '__main__':
    unittest.main()
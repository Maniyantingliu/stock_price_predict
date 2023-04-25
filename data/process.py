import pandas as pd
import csv
import xlrd
import argparse


def read_excel(file):
	wb = xlrd.open_workbook(filename=file)
	sheet1 = wb.sheet_by_index(0)
	# rows = sheet1.row_values(2)#获取行内容
	# cols = sheet1.col_values(2)#获取列内容
	# print(rows)
	# print(cols)
	# print(sheet1.cell(1,0).value)#获取表格里的内容，三种方式
	# print(sheet1.cell_value(1,0))
	# print(sheet1.row(1)[0].value)
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--stock_price_file", type=str)
	parser.add_argument("--stock_news_file", type=str)
	cfg = parser.parse_args()
	news_table = xlrd.open_workbook(filename=cfg.stock_news_file).sheet_by_index(0)
	stock_price_table = pd.read_csv(cfg.stock_price_file, sep=',')
	nrows = len(stock_price_table)-2
	new_data = []
	print(stock_price_table)
	for i in range(nrows,1,-1):
		stock_price = stock_price_table['收盘'][i]
		stock_news = news_table.cell_value(i-1, 0)
		date = stock_price_table['日期'][i]
		new_data.append([stock_news, stock_price, date])
		
	with open("news_price_file.csv", "w", encoding="utf-8", newline="") as f:
		csv_writer = csv.writer(f)
		name = ["news", "price", "date"]
		csv_writer.writerow(name)
		csv_writer.writerows(new_data)
		f.close

		
		
	
if __name__ == "__main__":
	main()
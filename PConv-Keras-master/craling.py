from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('created '+directory)
    except OSError:
        print('Error : already Created' + directory)

search = input('검색 ')
createFolder('data/'+search)
print('1')
url = f'https://www.google.co.kr/search?q={quote_plus(search)}&tbm=isch'
print("2")
driver = webdriver.Chrome("/Users/yunjeongyong/Downloads/chromedriver")
driver.get(url)
print("3")
options = webdriver.ChromeOptions()
print("4")
options.add_experimental_option('excludeSwitches', ['enable-logging'])
print("5")
driver.get(url)
print("6")
for i in range(500):
    driver.execute_script("window.scrollBy(0,50000)")
html = driver.page_source
print("7")
soup = BeautifulSoup(html, features="html.parser")
print("8")
img = soup.select('img')

n = 1
imgurl = []

for i in img:
    try:
        imgurl.append(i.attrs["src"])
    except KeyError:
        imgurl.append(i.attrs["data-src"])

for i in imgurl:
    urlretrieve(i, "data/"+ search +'/' + search + str(n) + ".jpg")
    n += 1
    print('downloading.........{}'.format(n))

driver.close()
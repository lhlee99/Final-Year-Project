import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import openpyxl
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import sys
import math
import re
import http.client
import urllib
import _thread
import shutil
import logging

popularUrl ='https://lihkg.com/category/5?order=now'
baseUrl = "https://lihkg.com"
LOGINTIME = 10
TIMEOUT = 3
FREQUENT = 4

def saveInJson(data,fileName):
    f = open(fileName, 'w', encoding='utf-8')
    f.write(json.dumps(data, ensure_ascii=False))
    f.close()
    print('Saved successfully!')

class PostScraper():
    def __init__(self):
        if sys.platform == 'linux':
            logging.basicConfig(filename='scrapping.log',level=logging.INFO, format='[%(asctime)s] %(name)s - %(levelname)s : %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s - %(levelname)s : %(message)s')

    def headDriver(self):
        options = Options()
        options.headless = False
        options.add_argument("--window-size=1920,1200")
        if sys.platform == 'linux':
            driver = webdriver.Chrome(options=options, executable_path="./chromedriver")
        else:
            driver = webdriver.Chrome(options=options, executable_path="chromedriver86.exe")
        return driver

    def headlessDriver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument(f"--window-size=1920, 900")
        chrome_options.add_argument("--hide-scrollbars")
        if sys.platform == 'linux':
            driver = webdriver.Chrome(options=chrome_options, executable_path="./chromedriver")
        else:
            driver = webdriver.Chrome(options=chrome_options, executable_path="./chromedriver86.exe")
        return driver

    def start(self):
        _thread.start_new_thread( self.scrape, () )

    def scrape(self, fre):
        FREQUENT = int(fre)
        now = datetime.now()
        nowStr = now.strftime("%Y%m%d-%H%M")
        targetFile = 'vm_dataset/' + nowStr + '.json' if sys.platform == 'linux' else 'data/' + nowStr + '.json'

        file = open(targetFile, 'w+', encoding='utf-8')

        logging.info("Start collecting data")
        toJson=[]
        driver = self.headlessDriver()
        driver.get(popularUrl)
        time.sleep(LOGINTIME)

        read_mores = driver.find_elements_by_xpath('//a[text()="Read More..."]')
        for read_more in read_mores:
            driver.execute_script("arguments[0].scrollIntoView();", read_more)
            driver.execute_script("$(arguments[0]).click();", read_more)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        flag = True
        while flag :
            time.sleep(1)
            recentList = driver.find_elements_by_xpath("//span[@class='_37XwjAqVHtjzqzEtybpHrU']")
            if len(recentList) == 0 :
                break
            else :
                driver.execute_script("arguments[0].scrollIntoView();", recentList[len(recentList) - 1 ] )

                if driver.find_elements_by_xpath("//span[@class='_37XwjAqVHtjzqzEtybpHrU']")[-2].text == "剛剛":
                    continue
                if len(driver.find_elements_by_xpath("//span[@class='_37XwjAqVHtjzqzEtybpHrU']")[-2].text.strip().split(" "))>=2:
                    if (int(driver.find_elements_by_xpath("//span[@class='_37XwjAqVHtjzqzEtybpHrU']")[-2].text.strip().split(" ")[0])>=FREQUENT and driver.find_elements_by_xpath("//span[@class='_37XwjAqVHtjzqzEtybpHrU']")[-2].text.strip().split(" ")[1]=="小時前"):
                    #if driver.find_elements_by_xpath("//span[@class='_37XwjAqVHtjzqzEtybpHrU']")[-2].text.strip().split(" ")[1]=="日前":
                        print("found")
                        flag = False
                else:
                    continue

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        postDoms = soup.find_all('div', attrs={'class':'wQ4Ran7ySbKd8PdMeHZZR'})
        print(len(postDoms))
        threadIdDoms = soup.find_all('a', attrs={'class':'_2A_7bGY9QAXcGu1neEYDJB'})
        threadIds = []

        for threadIdDom in threadIdDoms:
            alink = threadIdDom.attrs['href']
            threadId = alink.split("/thread/", 1)[1].split("/page", 1)[0]
            threadIds.append(threadId)
        clickNum= 0

        for postDom in postDoms:
            print(round(clickNum/len(postDoms),4))

            postName = postDom.find('span', attrs={'class':'_20jopXBFHNQ9FUbcGHLcHH'}).text
            postNetLike = 0
            if len(postDom.find_all('span', attrs={'class':'_37XwjAqVHtjzqzEtybpHrU'}))>=2:
                postNetLike = postDom.find_all('span', attrs={'class':'_37XwjAqVHtjzqzEtybpHrU'})[-1].text
            postPageNums = 1
            if postDom.find('div', attrs={'class':'_26oEXjfUS_iHzbxYcZE6bD'})!=None:
                postPageNums = int(postDom.find('div', attrs={'class':'_26oEXjfUS_iHzbxYcZE6bD'}).text.strip().split(" ")[0])
            post={}

            urlPost ="https://lihkg.com/thread/" + str(threadIds[clickNum]) + "/page/1"

            try:
                driver.get(urlPost)
                time.sleep(TIMEOUT)
                pageSoup1 = BeautifulSoup(driver.page_source, 'html.parser')
                commentDomForPT = pageSoup1.find('div', attrs={'class':'_36ZEkSvpdj_igmog0nluzh'})
            except TimeoutError:
                logging.exception("fail to get urlPost")
                continue

            if commentDomForPT == None:
                logging.info("Blocked from " + str(threadIds[clickNum]))
                logging.info(f"Post scrapped: ({clickNum}/{len(threadIds)})")
                idFile = 'threadId/' + nowStr + '.json' if sys.platform == 'linux' else 'ThreadId/' + nowStr + '.json'
                f = open(idFile, 'w+')
                f.write(json.dumps(threadIds[clickNum:]))
                f.close()
                logging.info(f"threadId File saved at {idFile}")
                break

            if commentDomForPT != None:

                post['threadId'] = threadIds[clickNum]
                post['postName'] = postName
                post['postNetLike'] = postNetLike

                postTime = datetime.now().strftime("%Y,%m.%d/ %H:%M:%S")
                postTime = postTime.replace(",","年")
                postTime = postTime.replace(".","月")
                postTime = postTime.replace("/","日")
        
                if commentDomForPT.find('span', attrs={'class':'Ahi80YgykKo22njTSCzs_'})!= None and len(commentDomForPT.find('span', attrs={'class':'Ahi80YgykKo22njTSCzs_'}).text.split(" "))>=2:
                    postTmp = commentDomForPT.find('span', attrs={'class':'Ahi80YgykKo22njTSCzs_'})
                    postTime = postTmp.attrs['title']
                post['postTime'] = postTime

                postContent = "Empty"
                if commentDomForPT.find('div', attrs={'class':'_2cNsJna0_hV8tdMj3X6_gJ'})!=None:
                    postContent= commentDomForPT.find('div', attrs={'class':'_2cNsJna0_hV8tdMj3X6_gJ'}).text
                elif commentDomForPT.find('div', attrs={'class':'oStuGCWyiP2IrBDndu1cY'})!=None:
                    postContent = commentDomForPT.find('div', attrs={'class':'oStuGCWyiP2IrBDndu1cY'}).text
                post['postContent'] = postContent

                postLike = 0
                postDislike = 0
                if len(commentDomForPT.find_all('label', attrs={'class':'_1yxPOd27pAzF9olhItRDej _2iRKJuMIV77zdwLRreUgLK'}))>=2:
                    postLike= commentDomForPT.find_all('label', attrs={'class':'_1yxPOd27pAzF9olhItRDej _2iRKJuMIV77zdwLRreUgLK'})[0].text
                    postDislike= commentDomForPT.find_all('label', attrs={'class':'_1yxPOd27pAzF9olhItRDej _2iRKJuMIV77zdwLRreUgLK'})[1].text
                post['postLike'] = int(postLike)
                post['postDislike'] = int(postDislike)

                comments=[]
                #Comments of each post
                skipCondition = False
                endCondition = False
                for i in range(postPageNums,1,-1):
                    urlPage="https://lihkg.com/thread/" + str(threadIds[clickNum]) + "/page/" + str(i)
                    try:
                        driver.get(urlPage)
                    except:
                        logging.exception("Fail to get urlPage")
                        continue

                    time.sleep(TIMEOUT)
                    pageSoup = BeautifulSoup(driver.page_source, 'html.parser')
                    commentDoms = pageSoup.find_all('div', attrs={'class':'_36ZEkSvpdj_igmog0nluzh'})
                    commentDoms.reverse()

                    #Comments of each page
                    position = 1
                    for commentDom in commentDoms:
                        comment={}
                        comTime = datetime.now().strftime("%Y,%m.%d/ %H:%M:%S")
                        comTime = comTime.replace(",","年")
                        comTime = comTime.replace(".","月")
                        comTime = comTime.replace("/","日")

                        if commentDom.find('span', attrs={'class':'Ahi80YgykKo22njTSCzs_'})!=None and len(commentDom.find('span', attrs={'class':'Ahi80YgykKo22njTSCzs_'}).text.split(" "))>=2:
                            commentTimeDay = commentDom.find('span', attrs={'class':'Ahi80YgykKo22njTSCzs_'}).text.split(" ")[1].strip()
                            commentBreak = int(commentDom.find('span', attrs={'class':'Ahi80YgykKo22njTSCzs_'}).text.split(" ")[0].strip())
                            
                            #FREQUENT hour
                            if commentBreak > FREQUENT and commentTimeDay == "小時前":
                                skipCondition = True
                            #1day
                            if commentTimeDay == "日前" or commentTimeDay == "個月前" or commentTimeDay == "年前":
                                skipCondition = True
                            #out of bound check
                            if i == postPageNums and position == 1 and commentBreak > FREQUENT and commentTimeDay == "小時前":
                                endCondition = True
                            #1day
                            if i == postPageNums and position == 1 and commentTimeDay == "日前" or commentTimeDay == "個月前" or commentTimeDay == "年前":
                                endCondition = True

                            commentTmp = commentDom.find('span', attrs={'class':'Ahi80YgykKo22njTSCzs_'})
                            comTime = commentTmp.attrs['title']

                        #break page
                        if skipCondition == True:
                            break
                        
                        commentId = "Empty"
                        if commentDom.find('span', attrs={'class': '_3SqN3KZ8m8vCsD9FNcxcki _208tAU6LsyjP5LKTdcPXD0'})!=None:
                            commentId = commentDom.find('span', attrs={'class': '_3SqN3KZ8m8vCsD9FNcxcki _208tAU6LsyjP5LKTdcPXD0'}).text.strip().split("#")[1]
                        elif commentDom.find('span', attrs={'class': '_3SqN3KZ8m8vCsD9FNcxcki'})!=None:
                            commentId = commentDom.find('span', attrs={'class': '_3SqN3KZ8m8vCsD9FNcxcki'}).text.strip().split("#")[1]
                        comment['commentId'] = commentId

                        commentWriter = "Empty"
                        if commentDom.find('span', attrs={'class':'ZZtOrmcIRcvdpnW09DzFk'})!=None:
                            commentWriter = commentDom.find('span', attrs={'class':'ZZtOrmcIRcvdpnW09DzFk'}).text
                        comment['writer'] = commentWriter

                        comment['time'] = comTime

                        commentContent = "Empty"
                        if commentDom.find('div', attrs={'class':'_2cNsJna0_hV8tdMj3X6_gJ'})!=None:
                            commentContent= commentDom.find('div', attrs={'class':'_2cNsJna0_hV8tdMj3X6_gJ'}).text
                        elif commentDom.find('div', attrs={'class':'oStuGCWyiP2IrBDndu1cY'})!=None:
                            commentContent = commentDom.find('div', attrs={'class':'oStuGCWyiP2IrBDndu1cY'}).text
                        comment['content'] = commentContent

                        commentLike = 0
                        commentDislike = 0
                        if len(commentDom.find_all('label', attrs={'class':'_1yxPOd27pAzF9olhItRDej _2iRKJuMIV77zdwLRreUgLK'}))>=2:
                            commentLike= commentDom.find_all('label', attrs={'class':'_1yxPOd27pAzF9olhItRDej _2iRKJuMIV77zdwLRreUgLK'})[0].text
                            commentDislike= commentDom.find_all('label', attrs={'class':'_1yxPOd27pAzF9olhItRDej _2iRKJuMIV77zdwLRreUgLK'})[1].text
                        comment['commentLike']= int(commentLike)
                        comment['commentDislike']= int(commentDislike)
                        comments.append(comment)

                        position += 1

                    #break post
                    if skipCondition == True:
                        skipCondition = False
                        break

                post['comments'] = comments
                toJson.append(post)
                saveInJson(toJson,targetFile)

            if endCondition == True:
                logging.info(f"Post scrapped: ({clickNum}/{len(threadIds)}) Remaining post out of bound: {len(threadIds)- clickNum}")
                break

            clickNum += 1

        logging.info(f"File saved at {targetFile}")

if __name__ == '__main__':
    #frequency = sys.argv[1]
    frequency = 4
    scraper = PostScraper()
    try:
        scraper.scrape(frequency)
    except:
        logging.exception("Failed to scrap")
    stream = os.popen('df -hT /dev/xvda1')
    logging.info(f'Remaining space:\n\t{stream.read()}')

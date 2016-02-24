from selenium import webdriver

from selenium.webdriver.common.keys import Keys

from time import sleep
browser = webdriver.Chrome()



print "opening"

browser.get("https://www.quora.com/Which-is-the-best-political-party-in-India-Why/log")

prev_y = -1
new_y = 0
while(prev_y != new_y):
	prev_y = new_y 
	end_ele = browser.find_element_by_class_name("LargeFooter")
	new_y = end_ele.location
	browser.execute_script("arguments[0].scrollIntoView();", end_ele)
	sleep(1)

u = browser.find_element_by_class_name('ContentWrapper')
s = open(browser.title+'.html','w')
s.write(u.get_attribute('innerHTML').encode('utf8'))
s.close()

from selenium import webdriver

#
driver = webdriver.Chrome()
driver.get("https://www.youtube.com/")
driver.minimize_window()
#driver.maximize_window()
#driver.quit()
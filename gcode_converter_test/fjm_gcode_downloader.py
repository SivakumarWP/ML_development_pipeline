import os
import time
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://fjm.wunderpen.com/device/7733fde7-6da5-4eee-9802-309009af28de/"
DOWNLOAD_DIR = "/Users/sivakumarvaradharajan/Downloads/gcode_by_section"
SCROLL_PASSES = 6

def setup_driver():
    opts = Options()
    # opts.add_argument("--headless=new")  # enable after you verify login once
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_window_size(1440, 960)
    return driver

def auto_scroll(driver, passes=6):
    last_h = 0
    for _ in range(passes):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.8)
        h = driver.execute_script("return document.body.scrollHeight")
        if h == last_h:
            break
        last_h = h

def collect_sections(driver):
    """
    Returns a list of {title: ..., links: [...]}, one per job section.
    """
    js = r"""
const result = [];
// Job cards usually have headings like "job from Campaign ..."
const cards = Array.from(document.querySelectorAll("div"))
  .filter(d => (d.textContent||"").trim().startsWith("job from Campaign"));

for (const headingDiv of cards) {
  const title = (headingDiv.textContent||"").trim().split("\n")[0];
  // assume the job card container is the heading plus its sibling container with links
  let cardContainer = headingDiv.closest("div");
  if (!cardContainer) continue;
  const gcodeLinks = Array.from(cardContainer.querySelectorAll('a[href$=".gcode"]'));
  if (gcodeLinks.length === 0) continue;
  const links = gcodeLinks.map(a => ({text:(a.textContent||"").trim(), href:a.href}));
  result.push({title:title, links:links});
}
return result;
"""
    return driver.execute_script(js)

def sanitize_filename(name: str) -> str:
    return "".join(ch for ch in name if ch not in r'\/:*?"<>|').strip()

def main():
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    driver = setup_driver()

    try:
        driver.get(URL)
        input("\nIf a login page appeared, sign in in the opened browser, "
              "navigate to the device page, then press Enter here to continue...")

        WebDriverWait(driver, 60).until(lambda d: "/device/" in d.current_url)
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        auto_scroll(driver, passes=SCROLL_PASSES)

        sections = collect_sections(driver)
        if not sections:
            print("No sections with .gcode files found.")
            return

        print(f"Found {len(sections)} job section(s).")

        sess = requests.Session()
        for c in driver.get_cookies():
            sess.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))

        base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(driver.current_url))

        for sec in sections:
            folder_name = sanitize_filename(sec["title"])[:80]
            folder_path = os.path.join(DOWNLOAD_DIR, folder_name)
            Path(folder_path).mkdir(parents=True, exist_ok=True)

            print(f"\n=== Section: {sec['title']} ===")
            for l in sec["links"]:
                url = l["href"] if l["href"].startswith("http") else urljoin(base, l["href"])
                name = l["text"] or os.path.basename(urlparse(url).path)
                safe = sanitize_filename(name)
                if not safe.lower().endswith(".gcode"):
                    safe += ".gcode"
                out_path = os.path.join(folder_path, safe)

                print(f"Downloading: {safe}")
                r = sess.get(url, stream=True)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 64):
                        if chunk:
                            f.write(chunk)
                print("Saved:", out_path)

    finally:
        time.sleep(1)
        driver.quit()

if __name__ == "__main__":
    main()

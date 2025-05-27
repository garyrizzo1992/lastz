import pygetwindow as gw

def list_all_window_titles():
    """Print all window titles."""
    titles = gw.getAllTitles()
    for title in titles:
        print(title)

if __name__ == "__main__":
    list_all_window_titles()
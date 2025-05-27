from log_tracker import Logger

def main():
    logger = Logger.get_logger()
    logger.info("Logger is ready!")



if __name__ == "__main__":
    main()

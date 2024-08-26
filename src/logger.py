import logging

class CustomLogger:
    """
    Custom logger class to provide consistent logging configuration across scripts.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize the CustomLogger.

        :param name: Name of the logger.
        :param level: Logging level (default is logging.INFO).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler and set level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        if not self.logger.hasHandlers():
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        Return the configured logger.

        :return: Configured logger.
        """
        return self.logger

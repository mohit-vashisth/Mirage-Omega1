from logging import LogRecord, Filter
from api.utils.context import get_request_id, get_ip, get_user_agent, get_path

class RequestContextFilter(Filter):
    def filter(self, record: LogRecord) -> bool:
        record.request_id = get_request_id()
        record.ip = get_ip()
        record.user_agent = get_user_agent()
        record.path = get_path()
        return True

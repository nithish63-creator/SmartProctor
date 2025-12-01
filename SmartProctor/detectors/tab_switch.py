import time
import platform

class TabSwitchMonitor:

    def __init__(self, logger=None):
        self.logger = logger
        self.last_state = 'active'  # 'active' or 'background'

    def _get_foreground_state(self):
        return 'active'

    def run_cycle(self):
        state = self._get_foreground_state()
        if state != self.last_state:
            # log state change
            if self.logger:
                self.logger.log('tab_switch', f'{self.last_state}->{state}')
            self.last_state = state
        # no-op otherwise
        return self.last_state

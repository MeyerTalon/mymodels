import sys

from core.config import load_config
from core.device import select_device
from core.training import parse_config_path
from stellaris.policy import PolicyConfig, load_policy, propose_action
from stellaris.screen import capture_screen


def main() -> None:
    config_path = parse_config_path('Propose a Stellaris action from the screen')
    try:
        settings = PolicyConfig.from_config(load_config(config_path))
    except (FileNotFoundError, TypeError, ValueError) as error:
        sys.exit(f'Error: {error}')
    policy = load_policy(settings, select_device())
    print(propose_action(policy, capture_screen(settings.screenshot_width_px)))


if __name__ == '__main__':
    main()

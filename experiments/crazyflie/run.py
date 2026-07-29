"""Crazyflie action dispatcher."""

from __future__ import annotations

from dataclasses import dataclass

from .components.utils import (
    PlanRepository,
    VALID_CONDITIONS,
    VALID_FANS,
    VALID_SCENARIOS,
)


@dataclass(frozen=True)
class RunRequest:
    action: str
    fan: int
    condition: str
    scenario: str
    plot: bool = False
    analysis: str = 'latest'
    run_number: int | None = None

    def __post_init__(self) -> None:
        valid = {
            'action': (self.action, ('plan', 'fly', 'analyze')),
            'fan': (self.fan, VALID_FANS),
            'condition': (self.condition, VALID_CONDITIONS),
            'scenario': (self.scenario, VALID_SCENARIOS),
            'analysis': (self.analysis, ('latest', 'all', 'summary')),
        }
        for name, (value, choices) in valid.items():
            if value not in choices:
                raise ValueError(f'{name} must be one of {choices}; got {value!r}.')
        if self.run_number is not None and self.run_number < 1:
            raise ValueError('run_number must be positive or None.')


class ExperimentRunner:
    def __init__(self, request: RunRequest) -> None:
        self.request = request

    def run(self) -> None:
        {'plan': self._plan, 'fly': self._fly, 'analyze': self._analyze}[self.request.action]()

    def _plan(self) -> None:
        from .components.planner import PlanningService

        PlanningService().run(
            fan=self.request.fan, scenario=self.request.scenario, plot=self.request.plot,
        )

    def _analyze(self) -> None:
        from .components.analysis import AnalysisService

        AnalysisService().run(
            condition=self.request.condition,
            fan=self.request.fan,
            scenario=self.request.scenario,
            mode=self.request.analysis,
            run_number=self.request.run_number,
        )

    def _fly(self) -> None:
        request = self.request
        if request.condition == 'pdstl':
            try:
                PlanRepository().require_flyable(request.fan, request.scenario)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise SystemExit(f'Refusing pdSTL flight: {exc}') from exc

        from ros_sugar import Launcher

        from .components.crazyflie import CrazyflieConfig, CrazyflieFlightComponent

        component = CrazyflieFlightComponent(
            component_name='crazyflie_experiment',
            config=CrazyflieConfig(
                condition=request.condition,
                fan_speed=request.fan,
                scenario=request.scenario,
            ),
        )
        launcher = Launcher()
        launcher.add_pkg(components=[component], activate_all_components_on_start=True)
        launcher.bringup()


def run_crazyflie_experiment(**selection) -> None:
    ExperimentRunner(RunRequest(**selection)).run()

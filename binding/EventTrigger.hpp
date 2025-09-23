#pragma once

#include <vector>
#include <functional>

struct EventTrigger {
    bool prev_state = false;
    bool triggered = false;
    double start_time = -1.0;

    double saved_time = 0.0;

    /**
     * @brief Contact Event Trigger
     * @param saved_values : Data at event trigger
    */
    std::vector<double> saved_values;   

    // update: rising edge 발생 시 데이터 캡처
    bool update(bool current_state, double sim_time,
                const std::function<std::vector<double>()>& capture) {
        bool event = false;
        if (current_state && !prev_state) {
            triggered = true;
            start_time = sim_time;
            saved_time = sim_time;
            saved_values = capture();   // ✅ capture 함수에서 원하는 데이터 벡터 반환
            event = true;
        }
        prev_state = current_state;
        return event;
    }

    // 경과 시간
    double elapsed(double sim_time) const {
        if (triggered && start_time >= 0)
            return sim_time - start_time;
        return 0;
    }
};

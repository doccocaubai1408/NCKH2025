# learn_logger.py
# ASCII-only.

import csv
import os
import time


class LearnLogger:
    def __init__(
        self,
        log_path="learn_log_5min.csv",
        summary_path="learn_summary_5min.txt",
        duration_sec=1800
    ):
        self.log_path = log_path
        self.summary_path = summary_path
        self.duration_sec = float(duration_sec)
        self.start_ts = time.time()

        self.learn_count = 0
        self.duration_sum = 0.0

        self.cpu_machine_cost_sum = 0.0
        self.cpu_total_cost_sum = 0.0

        self.current_A_sec_sum = 0.0
        self.current_active_duration_sum = 0.0

        self.total_infer_calls = 0

        self.summary_saved = False
        self.rows_ram = []

        self.header = [
            "time",
            "learn_index",
            "context_id",
            "context_light",
            "context_dark_q",
            "context_edge_q",
            "context_pin_bin",
            "learn_duration_sec",
            "model_count",
            "models_done",
            "avg_cpu_machine_percent",
            "avg_cpu_total_percent",
            "avg_current_A",
            "total_infer_calls"
        ]

    def active(self):
        return (time.time() - self.start_ts) <= self.duration_sec

    def log_context(self, context_id, context_obj, learn_duration_sec, model_rows):
        if not self.active():
            self.save_summary_once()
            return False

        if not model_rows:
            return False

        model_duration_sum = 0.0
        cpu_machine_cost = 0.0
        cpu_total_cost = 0.0
        current_cost = 0.0
        current_duration = 0.0

        current_vals = []
        total_calls = 0
        models_done = []

        for r in model_rows:
            models_done.append(str(r.get("model", "")))

            try:
                dur = float(r.get("duration", 0.0))
            except Exception:
                dur = 0.0

            if dur < 0.0:
                dur = 0.0

            model_duration_sum += dur

            cpu_machine = r.get("cpu_machine", None)
            if cpu_machine is not None:
                try:
                    cpu_machine_cost += float(cpu_machine) * dur
                except Exception:
                    pass

            cpu_total = r.get("cpu_total", None)
            if cpu_total is not None:
                try:
                    cpu_total_cost += float(cpu_total) * dur
                except Exception:
                    pass

            cur = r.get("current_A", None)
            if cur is not None:
                try:
                    cur_f = float(cur)
                    current_vals.append(cur_f)
                    current_cost += cur_f * dur
                    current_duration += dur
                except Exception:
                    pass

            try:
                total_calls += int(r.get("infer_calls", 0))
            except Exception:
                pass

        active_duration = model_duration_sum
        if active_duration <= 0.0:
            try:
                active_duration = float(learn_duration_sec)
            except Exception:
                active_duration = 0.0

        avg_cpu_machine = cpu_machine_cost / active_duration if active_duration > 0.0 else 0.0
        avg_cpu_total = cpu_total_cost / active_duration if active_duration > 0.0 else 0.0
        avg_current = current_cost / current_duration if current_duration > 0.0 else 0.0

        self.learn_count += 1
        self.duration_sum += float(active_duration)
        self.cpu_machine_cost_sum += float(cpu_machine_cost)
        self.cpu_total_cost_sum += float(cpu_total_cost)
        self.current_A_sec_sum += float(current_cost)
        self.current_active_duration_sum += float(current_duration)
        self.total_infer_calls += int(total_calls)

        if context_obj is None:
            context_obj = {}

        self.rows_ram.append([
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            int(self.learn_count),
            str(context_id),
            str(context_obj.get("light", "NA")),
            int(context_obj.get("dark_q", -1)),
            int(context_obj.get("edge_q", -1)),
            str(context_obj.get("pin_bin", "NA")),
            "%.6f" % float(active_duration),
            int(len(model_rows)),
            "|".join(models_done),
            "%.2f" % float(avg_cpu_machine),
            "%.2f" % float(avg_cpu_total),
            "%.6f" % float(avg_current),
            int(total_calls)
        ])

        return True

    def _write_log_file_once(self):
        tmp = self.log_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            for row in self.rows_ram:
                w.writerow(row)
        os.replace(tmp, self.log_path)

    def save_summary(self):
        elapsed = time.time() - self.start_ts

        avg_duration = self.duration_sum / self.learn_count if self.learn_count > 0 else 0.0

        active_ratio = self.duration_sum / elapsed if elapsed > 0.0 else 0.0

        avg_cpu_machine_active = (
            self.cpu_machine_cost_sum / self.duration_sum
            if self.duration_sum > 0.0 else 0.0
        )

        avg_cpu_total_active = (
            self.cpu_total_cost_sum / self.duration_sum
            if self.duration_sum > 0.0 else 0.0
        )

        avg_cpu_machine_whole_window = (
            self.cpu_machine_cost_sum / elapsed
            if elapsed > 0.0 else 0.0
        )

        avg_cpu_total_whole_window = (
            self.cpu_total_cost_sum / elapsed
            if elapsed > 0.0 else 0.0
        )

        avg_current_A_active = (
            self.current_A_sec_sum / self.current_active_duration_sum
            if self.current_active_duration_sum > 0.0 else 0.0
        )

        avg_current_A_whole_window = (
            self.current_A_sec_sum / elapsed
            if elapsed > 0.0 else 0.0
        )

        tmp = self.summary_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("learn summary\n")
            f.write("=============\n")
            f.write("elapsed_sec: %.2f\n" % float(elapsed))
            f.write("window_sec: %.2f\n" % float(self.duration_sec))
            f.write("learn_count: %d\n" % int(self.learn_count))
            f.write("active_learn_duration_sec: %.6f\n" % float(self.duration_sum))
            f.write("active_ratio: %.6f\n" % float(active_ratio))
            f.write("avg_learn_duration_sec: %.6f\n" % float(avg_duration))
            f.write("total_infer_calls: %d\n" % int(self.total_infer_calls))
            f.write("total_cpu_machine_percent_sec: %.6f\n" % float(self.cpu_machine_cost_sum))
            f.write("total_cpu_total_percent_sec: %.6f\n" % float(self.cpu_total_cost_sum))
            f.write("avg_cpu_machine_percent_active: %.2f\n" % float(avg_cpu_machine_active))
            f.write("avg_cpu_total_percent_active: %.2f\n" % float(avg_cpu_total_active))
            f.write("avg_cpu_machine_percent_whole_window: %.2f\n" % float(avg_cpu_machine_whole_window))
            f.write("avg_cpu_total_percent_whole_window: %.2f\n" % float(avg_cpu_total_whole_window))
            f.write("total_current_A_sec: %.6f\n" % float(self.current_A_sec_sum))
            f.write("avg_current_A_active: %.6f\n" % float(avg_current_A_active))
            f.write("avg_current_A_whole_window: %.6f\n" % float(avg_current_A_whole_window))
            f.write("log_file: %s\n" % str(self.log_path))
        os.replace(tmp, self.summary_path)

    def save_summary_once(self):
        if self.summary_saved:
            return

        if self.active():
            return

        try:
            self._write_log_file_once()
        except Exception as e:
            print("Learn log save failed:", str(e))

        try:
            self.save_summary()
        except Exception as e:
            print("Learn summary save failed:", str(e))

        self.summary_saved = True

    def close(self):
        if self.summary_saved:
            return

        try:
            self._write_log_file_once()
        except Exception as e:
            print("Learn log save failed:", str(e))

        try:
            self.save_summary()
        except Exception as e:
            print("Learn summary save failed:", str(e))

        self.summary_saved = True
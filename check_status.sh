#!/bin/bash
echo "================================================================================"
echo "  BACKTEST STATUS MONITOR"
echo "================================================================================"
echo ""

echo "QUICK TEST (7,200 sims):"
if pgrep -f "quick_regime_test.py" > /dev/null; then
    echo "  Status: RUNNING"
    tail -5 /tmp/claude/-root/tasks/b36b8d8.output 2>/dev/null | grep -E "(VOL REGIME|Running|Best)" || echo "  Initializing..."
else
    echo "  Status: COMPLETED or NOT RUNNING"
    if [ -f quick_test_results.json ]; then
        echo "  Results: quick_test_results.json ($(wc -c < quick_test_results.json) bytes)"
    fi
fi

echo ""
echo "FULL SWEEP (420,000 sims):"
if pgrep -f "regime_switching_backtest.py" > /dev/null; then
    echo "  Status: RUNNING"
    echo "  CPU usage: $(ps aux | grep 'regime_switching_backtest.py' | grep -v grep | awk '{print $3}')%"
    echo "  Runtime: $(ps -p $(pgrep -f regime_switching_backtest.py | head -1) -o etime= 2>/dev/null || echo 'N/A')"
    tail -5 ~/canonic_backtest/backtest_run.log 2>/dev/null | grep -E "(VOL REGIME|Progress|Best)" || echo "  Initializing..."
else
    echo "  Status: COMPLETED or NOT RUNNING"
    if [ -f backtest_results_full.json ]; then
        echo "  Results: backtest_results_full.json ($(wc -c < backtest_results_full.json) bytes)"
    fi
fi

echo ""
echo "OUTPUT FILES:"
ls -lh ~/canonic_backtest/*.json ~/canonic_backtest/*.md 2>/dev/null | awk '{print "  "$9" ("$5")"}'

echo ""
echo "================================================================================"

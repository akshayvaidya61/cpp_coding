#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <signal.signaler.h> // Include for timer notification

using namespace std::chrono_literals;

std::mutex mtx;
std::condition_variable cv;
bool running = true;

void timer_handler(int sig)
{
    std::lock_guard<std::mutex> lock(mtx);
    cv.notify_one();
}

void thread_func()
{
    while (running)
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []
                { return !running; }); // Wait for notification or stop flag
        if (!running)
        {
            break;
        }
        // Your code to be executed every 10 ms goes here
        std::cout << "Thread running...\n";
    }
}

int main()
{
    // Install timer signal handler
    struct sigevent sev;
    sev.sigev_notify = SIGEV_SIGNAL;
    sev.sigev_signo = SIGRTMIN;
    timer_create(CLOCK_REALTIME, &sev, nullptr);
    sigaction(SIGRTMIN, nullptr, reinterpret_cast<struct sigaction *>(&timer_handler));

    // Setup timer to fire every 10 milliseconds
    struct itimerspec spec;
    spec.it_value.tv_sec = 0;
    spec.it_value.tv_nsec = 10 * 1000 * 1000; // 10 milliseconds in nanoseconds
    spec.it_interval = spec.it_value;
    timer_settime(SIGRTMIN, 0, &spec);

    // Start the thread
    std::thread th(thread_func);

    // Main thread can do other work here...

    // Stop the thread
    {
        std::lock_guard<std::mutex> lock(mtx);
        running = false;
        cv.notify_one();
    }

    th.join();

    // Cleanup timer
    timer_delete(SIGRTMIN);

    return 0;
}

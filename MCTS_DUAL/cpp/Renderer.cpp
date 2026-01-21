#include "Renderer.h"
#include "IntersectionEnv.h"

#include <GLFW/glfw3.h>

#ifdef _WIN32
#define NOMINMAX
#define GLFW_EXPOSE_NATIVE_WIN32
#include <Windows.h>
#include <GLFW/glfw3native.h>
#endif

#include <cmath>
#include <iostream>
#include <array>
#include <cstdio>

struct Renderer::Impl {
    GLFWwindow* window{nullptr};
#ifdef _WIN32
    HWND hwnd{nullptr};
    HDC hdc{nullptr};
    HFONT font{nullptr};
    HICON hicon_small{nullptr};
    HICON hicon_big{nullptr};
    int fb_w{0};
    int fb_h{0};
#endif
};

// NDC mapping is based on fixed logical coordinates (WIDTH x HEIGHT).
// Window resizing is handled via glViewport letterbox/pillarbox.
static inline float ndc_x(float px){return px/(WIDTH*0.5f)-1.0f;}
static inline float ndc_y(float py){return 1.0f - py/(HEIGHT*0.5f);} // invert y

static void draw_rect_ndc(float x_px,float y_px,float w_px,float h_px, float r,float g,float b,float a=1.0f){
    float x0 = ndc_x(x_px);
    float y0 = ndc_y(y_px);
    float x1 = ndc_x(x_px + w_px);
    float y1 = ndc_y(y_px + h_px);
    glColor4f(r,g,b,a);
    glBegin(GL_QUADS);
    glVertex2f(x0,y0);
    glVertex2f(x1,y0);
    glVertex2f(x1,y1);
    glVertex2f(x0,y1);
    glEnd();
}

static void draw_line_px(float x0,float y0,float x1,float y1,float width, float r,float g,float b,float a=1.0f){
    glLineWidth(width);
    glColor4f(r,g,b,a);
    glBegin(GL_LINES);
    glVertex2f(ndc_x(x0), ndc_y(y0));
    glVertex2f(ndc_x(x1), ndc_y(y1));
    glEnd();
}

static void draw_circle_px(float cx,float cy,float radius,int segments,float r,float g,float b){
    glColor3f(r,g,b);
    glBegin(GL_TRIANGLE_FAN);
    for(int i=0;i<=segments;i++){
        constexpr float PI_F = 3.14159265358979323846f;
        float a = 2.0f * PI_F * float(i) / float(segments);
        float x = cx + std::cos(a)*radius;
        float y = cy + std::sin(a)*radius;
        glVertex2f(ndc_x(x), ndc_y(y));
    }
    glEnd();
}

Renderer::Renderer() {
    if(!init_glfw()) return;
    impl = std::make_unique<Impl>();
    // We use immediate-mode OpenGL (glBegin/glEnd). Core profile removes these APIs,
    // so we must request a COMPAT profile.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    impl->window = glfwCreateWindow(WIDTH*1.5, HEIGHT*1.5, "Intersection", nullptr, nullptr);
    if(!impl->window){
        glfwTerminate();
        return;
    }

#ifdef _WIN32
    // Set window icon from assets/icon.ico (place your ICO there; PNG not supported here)
    impl->hwnd = glfwGetWin32Window(impl->window);
    if(impl->hwnd){
        HICON icon_big = static_cast<HICON>(
            LoadImageW(nullptr, L"assets\\icon.ico", IMAGE_ICON, 0, 0,
                       LR_LOADFROMFILE | LR_DEFAULTSIZE)
        );
        HICON icon_small = static_cast<HICON>(
            LoadImageW(nullptr, L"assets\\icon.ico", IMAGE_ICON, 16, 16,
                       LR_LOADFROMFILE)
        );
        if(icon_big){
            SendMessageW(impl->hwnd, WM_SETICON, ICON_BIG, (LPARAM)icon_big);
            impl->hicon_big = icon_big;
        }
        if(icon_small){
            SendMessageW(impl->hwnd, WM_SETICON, ICON_SMALL, (LPARAM)icon_small);
            impl->hicon_small = icon_small;
        }
    }
#endif

    glfwMakeContextCurrent(impl->window);
    glfwSwapInterval(1);
    // No GLAD in this build: using immediate-mode OpenGL functions provided via system GL headers.
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

#ifdef _WIN32
    impl->hwnd = glfwGetWin32Window(impl->window);
    impl->hdc = nullptr;
    impl->font = CreateFontW(
        -18, 0, 0, 0,
        FW_NORMAL,
        FALSE, FALSE, FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"Segoe UI");
#endif

    initialized = true;
}

Renderer::~Renderer(){
#ifdef _WIN32
    if(impl){
        if(impl->hdc){
            ReleaseDC(impl->hwnd, impl->hdc);
            impl->hdc = nullptr;
        }
        if(impl->font){
            DeleteObject(impl->font);
            impl->font = nullptr;
        }
        if(impl->hicon_big){
            DestroyIcon(impl->hicon_big);
            impl->hicon_big = nullptr;
        }
        if(impl->hicon_small){
            DestroyIcon(impl->hicon_small);
            impl->hicon_small = nullptr;
        }
    }
#endif
    if(impl && impl->window){
        glfwDestroyWindow(impl->window);
        glfwTerminate();
    }
}

bool Renderer::init_glfw(){
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW" << std::endl;
        return false;
    }
    return true;
}

bool Renderer::window_should_close() const {
    if(!impl || !impl->window) return true;
    return glfwWindowShouldClose(impl->window) != 0;
}

void Renderer::poll_events() const {
    glfwPollEvents();
}

bool Renderer::key_pressed(int glfw_key) const {
    if(!impl || !impl->window) return false;
    return glfwGetKey(impl->window, glfw_key) == GLFW_PRESS;
}

void Renderer::render(const IntersectionEnv& env, bool show_lane_ids, bool show_lidar){
    if(!initialized) return;
    glfwMakeContextCurrent(impl->window);

    // Dynamic viewport (support window resizing) + keep aspect ratio (letterbox/pillarbox)
    int full_w=WIDTH, full_h=HEIGHT;
    glfwGetFramebufferSize(impl->window, &full_w, &full_h);
    const int view = (full_w < full_h) ? full_w : full_h;
    const int vp_x = (full_w - view) / 2;
    const int vp_y = (full_h - view) / 2;
    glViewport(vp_x, vp_y, view, view);

    glClearColor(34/255.f,139/255.f,34/255.f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    draw_road(env.num_lanes);
    draw_route(env);
    draw_cars(env);
    if(show_lidar) draw_lidar(env);

#ifndef _WIN32
    if(show_lane_ids) draw_lane_ids(env);
    draw_hud(env);
#endif

    glfwSwapBuffers(impl->window);

#ifdef _WIN32
    gdi_begin_frame(full_w, full_h);
    if(show_lane_ids) draw_lane_ids(env);
    draw_hud(env);
    gdi_end_frame();
#endif

    // Keep GLFW input responsive even if Python doesn't call poll_events()
    glfwPollEvents();
}

#ifdef _WIN32
static std::wstring to_wide(const std::string& s){
    if(s.empty()) return std::wstring();
    int needed = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if(needed <= 0){
        std::wstring out;
        out.reserve(s.size());
        for(unsigned char ch : s) out.push_back((wchar_t)ch);
        return out;
    }
    std::wstring w;
    w.resize((size_t)needed - 1);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), needed);
    return w;
}

void Renderer::gdi_begin_frame(int fb_w, int fb_h) const{
    if(!impl || !impl->hwnd) return;
    if(!impl->hdc) impl->hdc = GetDC(impl->hwnd);
    impl->fb_w = fb_w;
    impl->fb_h = fb_h;

    if(!impl->hdc) return;
    SetBkMode(impl->hdc, TRANSPARENT);
    if(impl->font) SelectObject(impl->hdc, impl->font);
}

void Renderer::gdi_draw_text_px(int x, int y, const std::string& text, unsigned int rgb) const{
    if(!impl || !impl->hdc) return;

    const COLORREF color = RGB((rgb >> 16) & 0xFFu, (rgb >> 8) & 0xFFu, rgb & 0xFFu);
    SetTextColor(impl->hdc, color);

    auto w = to_wide(text);
    TextOutW(impl->hdc, x, y, w.c_str(), (int)w.size());
}

void Renderer::gdi_end_frame() const{
    if(!impl || !impl->hwnd || !impl->hdc) return;
    ReleaseDC(impl->hwnd, impl->hdc);
    impl->hdc = nullptr;
}
#endif

// BITMAP FONT (HUD/LABELS) -------------------------------
#ifndef _WIN32
static void draw_glyph_px(float x, float y, const uint8_t* rows, int scale, float r, float g, float b, float a){
    const float s = float(scale);
    for(int row=0; row<BitmapFont8x8::H; ++row){
        uint8_t bits = rows[row];
        for(int col=0; col<BitmapFont8x8::W; ++col){
            const bool on = (bits & (0x80u >> col)) != 0;
            if(!on) continue;
            draw_rect_ndc(x + col*s, y + row*s, s, s, r, g, b, a);
        }
    }
}

void Renderer::draw_text_px(float x, float y, const std::string& text,
                            float r, float g, float b, float a,
                            int scale) const {
    float pen_x = x;
    for(char c: text){
        const uint8_t* g8 = BitmapFont8x8::glyph(c);
        draw_glyph_px(pen_x, y, g8, scale, r, g, b, a);
        pen_x += float(BitmapFont8x8::W * scale + scale); // 1px gap
    }
}

void Renderer::draw_label_px(float cx, float cy, const std::string& text,
                             float bg_r, float bg_g, float bg_b, float bg_a,
                             int scale) const {
    const int glyph_w = BitmapFont8x8::W * scale;
    const int glyph_h = BitmapFont8x8::H * scale;
    const int gap = scale;
    const int text_w = int(text.size()) * (glyph_w + gap) - gap;
    const int pad_x = 10;
    const int pad_y = 6;

    const float left = cx - 0.5f * float(text_w + pad_x);
    const float top  = cy - 0.5f * float(glyph_h + pad_y);

    draw_rect_ndc(left, top, float(text_w + pad_x), float(glyph_h + pad_y), bg_r, bg_g, bg_b, bg_a);
    draw_text_px(left + 0.5f*pad_x, top + 0.5f*pad_y, text, 1.0f, 1.0f, 1.0f, 1.0f, scale);
}

void Renderer::draw_lane_ids(const IntersectionEnv& env) const{
    // Match Intersection/env.py colors
    const float in_r = 0.0f, in_g = 0.0f, in_b = 200/255.f;
    const float out_r = 200/255.f, out_g = 0.0f, out_b = 0.0f;

    for(const auto& kv : env.lane_layout.points){
        const std::string& id = kv.first;
        const auto& p = kv.second;
        const bool is_in = id.rfind("IN_", 0) == 0;
        draw_label_px(p.first, p.second, id,
                      is_in ? in_r : out_r,
                      is_in ? in_g : out_g,
                      is_in ? in_b : out_b,
                      1.0f,
                      2);
    }
}

void Renderer::draw_hud(const IntersectionEnv& env) const{
    int agents_alive = 0;
    for(const auto& c: env.cars) if(c.alive) agents_alive++;

    std::string line = "STEP: " + std::to_string(env.step_count) + " | AGENTS: " + std::to_string(agents_alive);
    if(env.traffic_flow){
        line += " | TRAFFIC: " + std::to_string((int)env.traffic_cars.size());
    }
    if(!env.cars.empty() && env.cars[0].alive){
        float speed_ms = (env.cars[0].state.v * FPS) / SCALE;
        char buf[64];
        std::snprintf(buf, sizeof(buf), " | SPEED: %.1f M/S", speed_ms);
        line += buf;
    }

    draw_text_px(10.0f, 10.0f, line, 1.0f, 1.0f, 1.0f, 1.0f, 2);

    // Lidar debug line
    std::string lidar_line = "LIDAR: " + std::to_string((int)env.lidars.size());
    if(!env.lidars.empty()){
        const auto& lid = env.lidars[0];
        float min_d = lid.max_dist;
        float max_d = 0.0f;
        for(float d: lid.distances){
            if(d < min_d) min_d = d;
            if(d > max_d) max_d = d;
        }
        char buf2[256];
        std::snprintf(buf2, sizeof(buf2), " | RAYS: %d | MIN: %.1f | MAX: %.1f", (int)lid.distances.size(), min_d, max_d);
        lidar_line += buf2;


    }

    draw_text_px(10.0f, 34.0f, lidar_line, 1.0f, 1.0f, 1.0f, 1.0f, 2);
}

#else

void Renderer::draw_lane_ids(const IntersectionEnv& env) const{
    if(!impl || !impl->hdc) return;
    const unsigned int in_color = 0x0000C8;
    const unsigned int out_color = 0xC80000;

    // To match the old layout, we need to know the viewport transform
    const int view = (impl->fb_w < impl->fb_h) ? impl->fb_w : impl->fb_h;
    const int vp_x = (impl->fb_w - view) / 2;
    const int vp_y = (impl->fb_h - view) / 2;

    for(const auto& kv : env.lane_layout.points){
        const std::string& id = kv.first;
        const auto& p = kv.second;
        const bool is_in = id.rfind("IN_", 0) == 0;

        // Convert logical px to framebuffer px
        int fb_px = vp_x + int(p.first * view / WIDTH);
        int fb_py = vp_y + int(p.second * view / HEIGHT);

        // Center text
        SIZE text_size;
        auto wide_id = to_wide(id);
        GetTextExtentPoint32W(impl->hdc, wide_id.c_str(), (int)wide_id.size(), &text_size);
        fb_px -= text_size.cx / 2;
        fb_py -= text_size.cy / 2;

        gdi_draw_text_px(fb_px, fb_py, id, is_in ? in_color : out_color);
    }
}

void Renderer::draw_hud(const IntersectionEnv& env) const{
    int agents_alive = 0;
    for(const auto& c: env.cars) if(c.alive) agents_alive++;

    std::string line = "STEP: " + std::to_string(env.step_count) + " | AGENTS: " + std::to_string(agents_alive);
    if(env.traffic_flow){
        line += " | TRAFFIC: " + std::to_string((int)env.traffic_cars.size());
    }
    if(!env.cars.empty() && env.cars[0].alive){
        float speed_ms = (env.cars[0].state.v * FPS) / SCALE;
        char buf[64];
        std::snprintf(buf, sizeof(buf), " | SPEED: %.1f M/S", speed_ms);
        line += buf;
    }

    if(impl && impl->hdc){
        gdi_draw_text_px(10, 10, line, 0xFFFFFF);
    }

    // Lidar debug line
    std::string lidar_line = "LIDAR: " + std::to_string((int)env.lidars.size());
    if(!env.lidars.empty()){
        const auto& lid = env.lidars[0];
        float min_d = lid.max_dist;
        float max_d = 0.0f;
        for(float d: lid.distances){
            if(d < min_d) min_d = d;
            if(d > max_d) max_d = d;
        }
        char buf2[256];
        std::snprintf(buf2, sizeof(buf2), " | RAYS: %d | MIN: %.1f | MAX: %.1f", (int)lid.distances.size(), min_d, max_d);
        lidar_line += buf2;
    }

    if(impl && impl->hdc){
        gdi_draw_text_px(10, 34, lidar_line, 0xFFFFFF);
    }
}

#endif

// ROUTE ---------------------------------------------------
void Renderer::draw_route(const IntersectionEnv& env) const{
    if(env.cars.empty()) return;
    const auto& car = env.cars[0];
    if(car.path.empty()) return;

    // Match screenshot style: cyan route
    glLineWidth(2.0f);
    glColor4f(0.0f, 1.0f, 1.0f, 0.8f);
    glBegin(GL_LINE_STRIP);
    for(const auto& p : car.path){
        glVertex2f(ndc_x(p.first), ndc_y(p.second));
    }
    glEnd();

    // Lookahead target point (match Intersection agent observation)
    // lookahead = 10, target_idx = min(path_index + lookahead, len(path)-1)
    const int lookahead = 10;
    int target_idx = car.path_index + lookahead;
    if(target_idx < 0) target_idx = 0;
    if(target_idx >= (int)car.path.size()) target_idx = (int)car.path.size() - 1;

    const float tx = car.path[target_idx].first;
    const float ty = car.path[target_idx].second;

    // draw a small red dot
    draw_circle_px(tx, ty, 4.0f, 10, 1.0f, 0.0f, 0.0f);
}

// ROAD GEOMETRY -----------------------------------------
static void draw_center_lines(int num_lanes, float rw) {
    const float center_gap = 2.0f;
    const float cx = WIDTH * 0.5f;
    const float cy = HEIGHT * 0.5f;
    const float stop_off = rw + CORNER_RADIUS;

    // vertical yellow lines
    draw_line_px(cx - center_gap, 0, cx - center_gap, cy - stop_off, 2, 1.0f, 0.8f, 0.0f);
    draw_line_px(cx + center_gap, 0, cx + center_gap, cy - stop_off, 2, 1.0f, 0.8f, 0.0f);
    draw_line_px(cx - center_gap, HEIGHT, cx - center_gap, cy + stop_off, 2, 1.0f, 0.8f, 0.0f);
    draw_line_px(cx + center_gap, HEIGHT, cx + center_gap, cy + stop_off, 2, 1.0f, 0.8f, 0.0f);

    // horizontal yellow lines
    draw_line_px(0, cy - center_gap, cx - stop_off, cy - center_gap, 2, 1.0f, 0.8f, 0.0f);
    draw_line_px(0, cy + center_gap, cx - stop_off, cy + center_gap, 2, 1.0f, 0.8f, 0.0f);
    draw_line_px(WIDTH, cy - center_gap, cx + stop_off, cy - center_gap, 2, 1.0f, 0.8f, 0.0f);
    draw_line_px(WIDTH, cy + center_gap, cx + stop_off, cy + center_gap, 2, 1.0f, 0.8f, 0.0f);
}

static void draw_stop_lines(float rw) {
    const float cx = WIDTH * 0.5f;
    const float cy = HEIGHT * 0.5f;
    const float stop_off = rw + CORNER_RADIUS;
    const float w = 4.0f;

    draw_line_px(cx - rw, cy - stop_off, cx, cy - stop_off, w, 0.94f, 0.94f, 0.94f);
    draw_line_px(cx, cy + stop_off, cx + rw, cy + stop_off, w, 0.94f, 0.94f, 0.94f);
    draw_line_px(cx - stop_off, cy, cx - stop_off, cy + rw, w, 0.94f, 0.94f, 0.94f);
    draw_line_px(cx + stop_off, cy, cx + stop_off, cy - rw, w, 0.94f, 0.94f, 0.94f);
}

static void draw_lane_dashes(int num_lanes, float rw) {
    const float cx = WIDTH * 0.5f;
    const float cy = HEIGHT * 0.5f;
    const float stop_off = rw + CORNER_RADIUS;

    auto dash = [&](float x0, float y0, float x1, float y1) {
        float dist = std::hypot(x1 - x0, y1 - y0);
        const float dash_len = 20.0f;
        int steps = int(dist / (dash_len * 2));
        float dx = (x1 - x0) / dist;
        float dy = (y1 - y0) / dist;

        for (int i = 0; i <= steps; i++) {
            float sx = x0 + dx * i * dash_len * 2;
            float sy = y0 + dy * i * dash_len * 2;
            float ex = sx + dx * dash_len;
            float ey = sy + dy * dash_len;
            draw_line_px(sx, sy, ex, ey, 2, 0.94f, 0.94f, 0.94f);
        }
    };

    for (int i = 1; i < num_lanes; i++) {
        float off = i * LANE_WIDTH_PX;
        // vertical dashes
        dash(cx - off, 0, cx - off, cy - stop_off);
        dash(cx + off, 0, cx + off, cy - stop_off);
        dash(cx - off, HEIGHT, cx - off, cy + stop_off);
        dash(cx + off, HEIGHT, cx + off, cy + stop_off);
        // horizontal dashes
        dash(0, cy - off, cx - stop_off, cy - off);
        dash(0, cy + off, cx - stop_off, cy + off);
        dash(WIDTH, cy - off, cx + stop_off, cy - off);
        dash(WIDTH, cy + off, cx + stop_off, cy + off);
    }
}

void Renderer::draw_road(int num_lanes) const{
    float rw = num_lanes * LANE_WIDTH_PX;

    // Base road surface
    draw_rect_ndc(WIDTH * 0.5f - rw, 0, 2 * rw, HEIGHT, 60/255.f, 60/255.f, 60/255.f);
    draw_rect_ndc(0, HEIGHT * 0.5f - rw, WIDTH, 2 * rw, 60/255.f, 60/255.f, 60/255.f);

    // Rounded corner handling (match Intersection/env.py)
    const float cr = CORNER_RADIUS;
    const float cx = WIDTH * 0.5f;
    const float cy = HEIGHT * 0.5f;

    std::array<std::pair<float,float>,4> corner_squares={
        std::make_pair(cx - rw - cr, cy - rw - cr),
        std::make_pair(cx + rw,      cy - rw - cr),
        std::make_pair(cx - rw - cr, cy + rw),
        std::make_pair(cx + rw,      cy + rw)};
    for(auto p: corner_squares) draw_rect_ndc(p.first, p.second, cr, cr, 60/255.f,60/255.f,60/255.f);

    std::array<std::pair<float,float>,4> grass_centers={
        std::make_pair(cx - rw - cr, cy - rw - cr),
        std::make_pair(cx + rw + cr, cy - rw - cr),
        std::make_pair(cx - rw - cr, cy + rw + cr),
        std::make_pair(cx + rw + cr, cy + rw + cr)};
    for(auto c: grass_centers) draw_circle_px(c.first, c.second, cr, 32, 34/255.f,139/255.f,34/255.f);

    draw_center_lines(num_lanes, rw);
    draw_stop_lines(rw);
    draw_lane_dashes(num_lanes, rw);
}

// CAR DRAW -----------------------------------------------
void Renderer::draw_cars(const IntersectionEnv& env) const{
    auto draw_one=[&](const Car& car, float r,float g,float b, bool npc){
        if(!car.alive) return;
        float x=car.state.x; float y=car.state.y; float heading=car.state.heading;
        float len=CAR_LENGTH; float wid=CAR_WIDTH;

        float hl=len*0.5f; float hw=wid*0.5f;

        auto rot=[&](float lx,float ly){
            float vx = lx * std::cos(-heading) - ly * std::sin(-heading);
            float vy = lx * std::sin(-heading) + ly * std::cos(-heading);
            return std::pair<float,float>(x+vx, y+vy);
        };

        // Body
        glColor3f(r,g,b);
        std::array<std::pair<float,float>,4> body={{
            rot(+hl,+hw), rot(+hl,-hw), rot(-hl,-hw), rot(-hl,+hw)
        }};
        glBegin(GL_QUADS);
        for(const auto& p: body){ glVertex2f(ndc_x(p.first), ndc_y(p.second)); }
        glEnd();

        // Head marker rectangle (matches Intersection/env.py)
        float mr = npc ? 0.0f : 200/255.f;
        float mg = npc ? 0.0f : 200/255.f;
        float mb = npc ? 0.0f : 200/255.f;
        glColor3f(mr,mg,mb);

        float x0 = -hl + 0.70f*len;
        float x1 = -hl + 0.95f*len;
        float y0 = -hw + 2.0f;
        float y1 = +hw - 2.0f;

        std::array<std::pair<float,float>,4> head={{
            rot(x0,y0), rot(x1,y0), rot(x1,y1), rot(x0,y1)
        }};
        glBegin(GL_QUADS);
        for(const auto& p: head){ glVertex2f(ndc_x(p.first), ndc_y(p.second)); }
        glEnd();
    };

    static const std::array<std::array<float,3>,6> colors={{
        {231/255.f,76/255.f,60/255.f},{52/255.f,152/255.f,219/255.f},{46/255.f,204/255.f,113/255.f},
        {155/255.f,89/255.f,182/255.f},{241/255.f,196/255.f,15/255.f},{230/255.f,126/255.f,34/255.f}}};

    // Ego/agents
    for(size_t idx=0; idx<env.cars.size(); ++idx){
        auto col = colors[idx%colors.size()];
        draw_one(env.cars[idx], col[0], col[1], col[2], false);
    }

    // Traffic NPCs: gray body + black head marker
    for(const auto& npc : env.traffic_cars){
        draw_one(npc, 150/255.f, 150/255.f, 150/255.f, true);
    }
}

// LIDAR ---------------------------------------------------
void Renderer::draw_lidar(const IntersectionEnv& env) const{
    const float line_r = 0.0f;
    const float line_g = 1.0f;
    const float line_b = 0.0f;
    const float line_a = 0.35f;

    const float hit_r = 1.0f;
    const float hit_g = 0.0f;
    const float hit_b = 0.0f;

    // Match Intersection/sensor.py: draw only hit rays
    const bool draw_all = false;

    for(size_t i=0;i<env.cars.size() && i<env.lidars.size();++i){
        if(!env.cars[i].alive) continue;
        const auto &lid=env.lidars[i];
        const auto &car=env.cars[i];
        float cx=car.state.x; float cy=car.state.y; float heading=car.state.heading;
        for(size_t k=0;k<lid.distances.size();++k){
            float dist=lid.distances[k];
            const bool hit = dist < lid.max_dist - 0.1f;
            if(!draw_all && !hit) continue;

            float ang=heading + lid.rel_angles[k];
            float ex=cx + dist*std::cos(ang);
            float ey=cy - dist*std::sin(ang);

            draw_line_px(cx,cy,ex,ey,2.0f,line_r,line_g,line_b,line_a);
            if(hit){
                draw_circle_px(ex,ey,2.0f,6,hit_r,hit_g,hit_b);
            }
        }
    }
}

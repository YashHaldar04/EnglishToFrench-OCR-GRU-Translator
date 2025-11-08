import cv2
import numpy as np
from collections import deque

# ---------- Helpers ----------
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))

def line_intersection(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (int(px), int(py))

def rescale_pts(pts, scale_x, scale_y):
    return np.array([[p[0]*scale_x, p[1]*scale_y] for p in pts], dtype="float32")

# ---------- Detection pipelines ----------
def detect_with_color_mask(proc):
    # proc = resized color image (BGR)
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    # white mask: low saturation, high value
    lower = np.array([0, 0, 180])
    upper = np.array([179, 70, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # morph close then open to fill holes
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.15 * proc.shape[0] * proc.shape[1]:
        return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:
        return approx.reshape(4,2)
    # fallback: convex hull approx to 4 points
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02*peri, True)
    if len(approx) == 4:
        return approx.reshape(4,2)
    return None

def detect_with_edges(proc):
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.12 * proc.shape[0] * proc.shape[1]:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def detect_with_hough(proc):
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=proc.shape[1]//6, maxLineGap=30)
    if lines is None:
        return None
    lines = lines[:,0,:]  # Nx4
    # split roughly vertical and horizontal
    vert, hor = [], []
    for x1,y1,x2,y2 in lines:
        angle = abs(np.degrees(np.arctan2((y2-y1),(x2-x1))))
        if angle > 45:
            vert.append((x1,y1,x2,y2))
        else:
            hor.append((x1,y1,x2,y2))
    if not vert or not hor:
        return None
    # intersections
    pts = []
    for v in vert:
        for h in hor:
            p = line_intersection(v, h)
            if p is not None and 0 <= p[0] < proc.shape[1] and 0 <= p[1] < proc.shape[0]:
                pts.append(p)
    if len(pts) < 4:
        return None
    pts = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02*peri, True)
    if len(approx) == 4:
        return approx.reshape(4,2)
    # try to pick 4 extreme points
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    quad = np.array([tl,tr,br,bl], dtype="float32")
    return quad

def detect_page_quad(frame_resized):
    # Try color mask first (white paper)
    q = detect_with_color_mask(frame_resized)
    if q is not None:
        return q
    # try edges based detection
    q = detect_with_edges(frame_resized)
    if q is not None:
        return q
    # try Hough-based detection
    q = detect_with_hough(frame_resized)
    return q

# ---------- Main scanner ----------
def pick_camera_index(max_index=3):
    for i in range(max_index+1):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        cap.release()
        if ok:
            return i
    return 0

def run_scanner(auto_capture=True, stable_frames=8, process_width=1000):
    cam_index = pick_camera_index(3)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Could not open camera. Check permissions and index.")
        return

    print("Scanner running. Align white page with black text. (ESC to quit)")
    stable_count = 0
    last_quad = None
    recent_quads = deque(maxlen=stable_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        orig_h, orig_w = frame.shape[:2]
        scale = 1.0
        if orig_w > process_width:
            scale = process_width / orig_w
            proc = cv2.resize(frame, (int(orig_w*scale), int(orig_h*scale)))
        else:
            proc = frame.copy()

        quad = detect_page_quad(proc)  # coordinates on proc
        if quad is not None:
            # scale quad back to original frame coordinates
            scale_x = orig_w / proc.shape[1]
            scale_y = orig_h / proc.shape[0]
            quad_orig = rescale_pts(quad, scale_x, scale_y).astype(int)
            recent_quads.append(quad_orig)
            # compute stability by comparing last few quads
            if last_quad is not None:
                # measure average corner distance
                d = np.mean(np.linalg.norm(quad_orig - last_quad, axis=1))
                if d < 20:  # small movement
                    stable_count += 1
                else:
                    stable_count = 0
            else:
                stable_count = 0
            last_quad = quad_orig
        else:
            recent_quads.append(None)
            last_quad = None
            stable_count = 0

        display = frame.copy()
        if last_quad is not None:
            pts = last_quad.reshape((-1,1,2))
            cv2.polylines(display, [pts], True, (0,255,255), 3)  # yellow
            for (x,y) in last_quad:
                cv2.circle(display, (int(x),int(y)), 6, (0,255,255), -1)

            cv2.putText(display, f"Stable: {stable_count}/{stable_frames}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        else:
            cv2.putText(display, "No page detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow("Crazy Scanner Preview", display)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        if key == 32:  # SPACE manual capture
            if last_quad is not None:
                warped = four_point_transform(frame, last_quad)
                cv2.imwrite("scanned_page.png", warped)
                cv2.imwrite("scanned_page_color.png", warped)
                print("✅ Saved scanned_page.png")
            else:
                print("⚠️ No page detected to capture.")
        # Auto-capture when stable
        if auto_capture and stable_count >= stable_frames and last_quad is not None:
            # average the last few quads to reduce jitter
            quads = [q for q in recent_quads if q is not None]
            if quads:
                avg = np.mean(np.stack(quads), axis=0)
                warped = four_point_transform(frame, avg)
                # enhance contrast a bit (convert to grayscale + adaptive threshold copy)
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                # save both color and B/W improved
                cv2.imwrite("scanned_page.png", warped)
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite("scanned_page_bw.png", bw)
                print("✅ Auto-saved scanned_page.png and scanned_page_bw.png")
                # give a little pause to avoid multiple saves
                cv2.putText(display, "Captured!", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
                cv2.imshow("Crazy Scanner Preview", display)
                cv2.waitKey(800)
                stable_count = 0
                recent_quads.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_scanner(auto_capture=True, stable_frames=8, process_width=900)
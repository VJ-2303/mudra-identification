import cv2
import mediapipe as mp
import math

# ============================================================
# CONFIG
# ============================================================
DEBUG = False          # Set True to print debug info on frame
CAMERA_INDEX = 1       # Change if your webcam index is different

# ============================================================
# Helpers: distance / normalization / precompute
# ============================================================
def get_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def get_scale_ref(landmarks):
    """Palm width: wrist (0) to middle_mcp (9)."""
    return get_distance(landmarks[0], landmarks[9]) + 1e-6


def compute_distance_tables(landmarks):
    """
    Compute raw distances and normalized distances between all landmark pairs.
    Returns (dist_table, ndist_table, scale_ref)
    dist_table keys: (i, j) with i < j
    """
    scale_ref = get_scale_ref(landmarks)
    dist = {}
    ndist = {}
    for i in range(21):
        for j in range(i + 1, 21):
            d = math.sqrt(
                (landmarks[i].x - landmarks[j].x) ** 2 +
                (landmarks[i].y - landmarks[j].y) ** 2
            )
            dist[(i, j)] = d
            ndist[(i, j)] = d / scale_ref
    return dist, ndist, scale_ref


def _key(i, j):
    return (i, j) if i < j else (j, i)


def raw_dist(i, j, dist_table):
    return dist_table.get(_key(i, j), 0.0)


def norm_dist(i, j, ndist_table):
    return ndist_table.get(_key(i, j), 0.0)


def nd(i, j, ndist_table):
    """Short alias for normalized distance."""
    return norm_dist(i, j, ndist_table)


# ============================================================
# Angle helpers
# ============================================================
def angle_between(v1, v2):
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return None
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    cos_a = max(-1.0, min(1.0, cos_a))  # clamp
    return math.degrees(math.acos(cos_a))


def get_angle(a, b, c):
    """
    Returns angle (in degrees) at point b formed by points a-b-c.
    Works directly with mediapipe landmark objects having .x and .y
    """
    try:
        ax, ay = a.x, a.y
        bx, by = b.x, b.y
        cx, cy = c.x, c.y

        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return 180

        cosA = dot / (mag1 * mag2)
        cosA = max(-1, min(1, cosA))
        return math.degrees(math.acos(cosA))
    except Exception:
        return 180


# ============================================================
# Straightness check (normalized)
# ============================================================
def is_finger_straight(landmarks, dist_table, ndist_table,
                       mcp_index, pip_index, tip_index,
                       threshold=0.9):
    """
    straightness_ratio = (MCP->TIP) / (MCP->PIP + PIP->TIP)
    Uses normalized distances (divided by palm width).
    """
    mcp_pip = norm_dist(mcp_index, pip_index, ndist_table)
    pip_tip = norm_dist(pip_index, tip_index, ndist_table)
    mcp_tip = norm_dist(mcp_index, tip_index, ndist_table)

    total = mcp_pip + pip_tip
    if total == 0:
        return False
    straightness_ratio = mcp_tip / total
    return straightness_ratio > threshold


# ============================================================
# Mudra functions
# ============================================================
def is_pataka_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, 0.97)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.97)
    ring_straight = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.97)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.97)
    if not (index_straight and middle_straight and ring_straight and pinky_straight):
        return False
    thumb_index_nd = norm_dist(4, 5, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    thumb_tucked = thumb_index_nd < (mcp_nd * 1.5)
    return thumb_tucked


def is_tripataka_mudra(landmarks, dist, ndist, scale):
    
    # 1) Index, middle, pinky straight
    if not is_finger_straight(landmarks, dist, ndist, 5,6,8,0.90): return False
    if not is_finger_straight(landmarks, dist, ndist, 9,10,12,0.90): return False
    if not is_finger_straight(landmarks, dist, ndist, 17,18,20,0.90): return False

    # 2) Ring must be bent
    if is_finger_straight(landmarks, dist, ndist, 13,14,16,0.88):
        return False

    # 3) Thumb MUST NOT touch ring tip (important to avoid Mayura conflict)
    if norm_dist(4, 16, ndist) < scale * 0.9:
        return False

    # 4) Thumb should be tucked near the palm (Tripataka style)
    if norm_dist(4, 5, ndist) > scale * 1.6:
        return False

    return True



def is_mayura_mudra(landmarks, dist, ndist, scale):
    
    # 1) Index, middle, pinky should be straight
    if not is_finger_straight(landmarks, dist, ndist, 5,6,8,0.90): return False
    if not is_finger_straight(landmarks, dist, ndist, 9,10,12,0.90): return False
    if not is_finger_straight(landmarks, dist, ndist, 17,18,20,0.90): return False

    # 2) Ring MUST be bent
    if is_finger_straight(landmarks, dist, ndist, 13,14,16,0.88):
        return False

    # 3) Thumb tip MUST TOUCH ring tip (strict)
    if norm_dist(4, 16, ndist) > scale * 0.85:
        return False

    return True



def is_ardha_chandra_mudra(landmarks, dist_table, ndist_table, scale_ref):
    thumb_straight = is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, 0.85)
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    ring_straight = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20)
    if not (thumb_straight and index_straight and middle_straight and ring_straight and pinky_straight):
        return False

    thumb_mcp = landmarks[2]
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    index_tip = landmarks[8]
    v_thumb = (thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y)
    v_index = (index_tip.x - index_mcp.x, index_tip.y - index_mcp.y)
    ang = angle_between(v_thumb, v_index)
    if ang is None:
        return False
    if not (60 <= ang <= 120):
        return False

    index_middle_nd = norm_dist(8, 12, ndist_table)
    middle_ring_nd = norm_dist(12, 16, ndist_table)
    ring_pinky_nd = norm_dist(16, 20, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    threshold = mcp_nd * 1.5
    return (index_middle_nd < threshold and
            middle_ring_nd < threshold and
            ring_pinky_nd < threshold)


def is_trishula_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    ring_straight = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16)
    if not (index_straight and middle_straight and ring_straight):
        return False
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.8)
    if not pinky_bent:
        return False
    thumb_pinky_nd = norm_dist(4, 20, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    return thumb_pinky_nd < (mcp_nd * 1.0)


def is_arala_mudra(landmarks, dist_table, ndist_table, scale_ref):
    """
    Arala Mudra:
      - Middle, ring, pinky: very straight and close
      - Index: clearly bent
      - Thumb: straight
    """
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, 0.85)
    if index_straight:
        return False

    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.97)
    ring_straight = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.97)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.97)
    if not (middle_straight and ring_straight and pinky_straight):
        return False

    d_mr = norm_dist(12, 16, ndist_table)
    d_rp = norm_dist(16, 20, ndist_table)
    d_mp = norm_dist(12, 20, ndist_table)
    close_thresh = scale_ref * 1.2
    if not (d_mr < close_thresh and d_rp < close_thresh and d_mp < close_thresh):
        return False

    idx_pip = get_angle(landmarks[5], landmarks[6], landmarks[7])
    idx_dip = get_angle(landmarks[6], landmarks[7], landmarks[8])
    if idx_pip > 165 and idx_dip > 165:
        return False

    thumb_straight = is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, 0.93)
    if not thumb_straight:
        return False

    return True


def is_kartari_mukham_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    if not (index_straight and middle_straight):
        return False

    index_tip_wrist_nd = norm_dist(8, 0, ndist_table)
    index_mcp_wrist_nd = norm_dist(5, 0, ndist_table)
    middle_tip_wrist_nd = norm_dist(12, 0, ndist_table)
    middle_mcp_wrist_nd = norm_dist(9, 0, ndist_table)
    if not (index_tip_wrist_nd > index_mcp_wrist_nd and
            middle_tip_wrist_nd > middle_mcp_wrist_nd):
        return False

    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.8)
    if not (ring_bent and pinky_bent):
        return False

    thumb_ring_pip_nd = norm_dist(4, 14, ndist_table)
    thumb_ring_tip_nd = norm_dist(4, 16, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    touch_threshold = mcp_nd * 2.0
    return (thumb_ring_pip_nd < touch_threshold) or (thumb_ring_tip_nd < touch_threshold)


def is_shuka_tundam_mudra(landmarks, dist_table, ndist_table, scale_ref):
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20)
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, 0.85)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.85)
    if not (middle_straight and pinky_straight and index_bent and ring_bent):
        return False
    thumb_index_nd = norm_dist(4, 5, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    return thumb_index_nd < (mcp_nd * 1.5)


def is_musthi_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, 0.8)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.8)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.8)
    if not (index_bent and middle_bent and ring_bent and pinky_bent):
        return False
    thumb_index_pip_nd = norm_dist(4, 6, ndist_table)
    thumb_middle_pip_nd = norm_dist(4, 10, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    touch_threshold = mcp_nd * 1.5
    return (thumb_index_pip_nd < touch_threshold) or (thumb_middle_pip_nd < touch_threshold)


def is_shikharam_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, 0.8)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.8)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.8)
    if not (index_bent and middle_bent and ring_bent and pinky_bent):
        return False
    thumb_straight = is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, 0.85)
    if not thumb_straight:
        return False
    thumb_index_nd = norm_dist(4, 6, ndist_table)
    thumb_middle_nd = norm_dist(4, 10, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    touch_threshold = mcp_nd * 1.5
    thumb_is_tucked = (thumb_index_nd < touch_threshold or thumb_middle_nd < touch_threshold)
    return not thumb_is_tucked


def is_chandrakala_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    thumb_straight = is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, 0.85)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.8)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.8)
    if not (index_straight and thumb_straight and middle_bent and ring_bent and pinky_bent):
        return False
    v_thumb = (landmarks[4].x - landmarks[2].x, landmarks[4].y - landmarks[2].y)
    v_index = (landmarks[8].x - landmarks[5].x, landmarks[8].y - landmarks[5].y)
    ang = angle_between(v_thumb, v_index)
    if ang is None:
        return False
    return ang > 45


def is_suchi_mudra(landmarks, dist_table, ndist_table, scale_ref):
    """
    Strict SUCHI:
      - Index very straight and extended
      - Other fingers bent
      - Thumb touches middle or ring (not index)
    """
    pip = get_angle(landmarks[5], landmarks[6], landmarks[7])
    dip = get_angle(landmarks[6], landmarks[7], landmarks[8])
    if not (pip > 170 and dip > 170):
        return False

    wrist = landmarks[0]
    tip = landmarks[8]
    mcp = landmarks[5]
    if math.dist((tip.x, tip.y), (wrist.x, wrist.y)) <= math.dist((mcp.x, mcp.y), (wrist.x, wrist.y)):
        return False

    if is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.75):
        return False
    if is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.75):
        return False
    if is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.75):
        return False

    ref = nd(5, 9, ndist_table)
    if ref == 0:
        return False

    t_mid1 = nd(4, 10, ndist_table)
    t_mid2 = nd(4, 11, ndist_table)
    t_ring1 = nd(4, 14, ndist_table)
    t_ring2 = nd(4, 15, ndist_table)
    touch_thresh = ref * 1.6
    touches_mid_ring = (
        t_mid1 < touch_thresh or
        t_mid2 < touch_thresh or
        t_ring1 < touch_thresh or
        t_ring2 < touch_thresh
    )
    if not touches_mid_ring:
        return False

    if nd(4, 8, ndist_table) < ref * 2.0:
        return False

    return True


def is_ardhapataka_mudra(landmarks, dist_table, ndist_table, scale_ref):
    index_straight = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8)
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12)
    if not (index_straight and middle_straight):
        return False
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.8)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.8)
    if not (ring_bent and pinky_bent):
        return False
    thumb_index_nd = norm_dist(4, 5, ndist_table)
    mcp_nd = norm_dist(5, 9, ndist_table)
    return thumb_index_nd < (mcp_nd * 1.5)


def is_kapitha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    try:
        if is_mukula_mudra(landmarks, dist_table, ndist_table, scale_ref):
            return False
    except Exception:
        pass

    ref = nd(5, 9, ndist_table)
    if ref == 0:
        return False

    t_ip = nd(4, 6, ndist_table)
    t_id = nd(4, 7, ndist_table)
    t_it = nd(4, 8, ndist_table)
    contact_thresh = ref * 2.2
    touching_index = (t_ip < contact_thresh or t_id < contact_thresh or t_it < contact_thresh)
    if not touching_index:
        return False

    pip_angle = get_angle(landmarks[5], landmarks[6], landmarks[7])
    dip_angle = get_angle(landmarks[6], landmarks[7], landmarks[8])
    if pip_angle > 165 and dip_angle > 165:
        return False

    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.90)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.90)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.90)
    if not (middle_bent and ring_bent and pinky_bent):
        return False

    return True


def is_katakamukha_mudra(landmarks, dist, ndist, scale):
    
    # ---- 1. Ring & Pinky must be straight ----
    ring_straight  = is_finger_straight(landmarks, dist, ndist, 13, 14, 16, threshold=0.82)
    pinky_straight = is_finger_straight(landmarks, dist, ndist, 17, 18, 20, threshold=0.82)

    if not (ring_straight and pinky_straight):
        return False

    # ---- 2. Index + Middle must be slightly bent (NOT straight) ----
    index_straight = is_finger_straight(landmarks, dist, ndist, 5, 6, 8, threshold=0.90)
    middle_straight = is_finger_straight(landmarks, dist, ndist, 9, 10, 12, threshold=0.90)

    if index_straight or middle_straight:
        return False  # they cannot be fully straight

    # ---- 3. Thumb must touch index OR middle ----
    thumb_index = nd(4, 8, ndist)
    thumb_middle = nd(4, 12, ndist)

    touch_thresh = scale * 1.6  # relaxed

    if not (thumb_index < touch_thresh or thumb_middle < touch_thresh):
        return False

    # ---- 4. Thumb MUST NOT touch ring/pinky (to avoid Hamsasya confusion) ----
    if nd(4, 16, ndist) < scale * 1.2:
        return False
    if nd(4, 20, ndist) < scale * 1.2:
        return False

    return True




def is_padmakosha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    """
    Padmakosha: curved fingers forming a flower bowl.
    """
    straight_flags = [
        is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, 0.93),
        is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.93),
        is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.93),
        is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.93),
    ]
    if sum(straight_flags) >= 3:
        return False

    mcp_points = [(landmarks[i].x, landmarks[i].y) for i in [5, 9, 13, 17]]
    palm_center = (
        sum(p[0] for p in mcp_points) / 4,
        sum(p[1] for p in mcp_points) / 4,
    )

    thumb_tip = (landmarks[4].x, landmarks[4].y)
    thumb_dist = math.dist(thumb_tip, palm_center)
    if not (scale_ref * 0.8 < thumb_dist < scale_ref * 4.0):
        return False

    tip_ids = [8, 12, 16, 20]
    tips = [(landmarks[i].x, landmarks[i].y) for i in tip_ids]
    xs = [p[0] for p in tips]
    ys = [p[1] for p in tips]
    if (max(xs) - min(xs)) < scale_ref * 0.4 and (max(ys) - min(ys)) < scale_ref * 0.4:
        return False

    d_im = norm_dist(8, 12, ndist_table)
    d_mr = norm_dist(12, 16, ndist_table)
    d_rp = norm_dist(16, 20, ndist_table)
    if not (scale_ref * 0.3 < d_im < scale_ref * 3.2 and
            scale_ref * 0.3 < d_mr < scale_ref * 3.2 and
            scale_ref * 0.3 < d_rp < scale_ref * 3.2):
        return False

    def curved(mcp, tip):
        d_mcp_tip = norm_dist(mcp, tip, ndist_table)
        d_wrist_mcp = norm_dist(0, mcp, ndist_table)
        return d_mcp_tip < d_wrist_mcp * 1.7

    if not (curved(5, 8) and curved(9, 12) and curved(13, 16) and curved(17, 20)):
        return False

    return True


def is_sarpashirsha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    loose = 0.80
    idx = is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, loose)
    mid = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, loose)
    rng = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, loose)
    pnk = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, loose)
    if not (idx and mid and rng and pnk):
        return False
    tip_dist = norm_dist(8, 20, ndist_table)
    mcp_dist = norm_dist(5, 17, ndist_table)
    if not (tip_dist < mcp_dist):
        return False
    thumb_index_nd = norm_dist(4, 5, ndist_table)
    mcp_ref_nd = norm_dist(5, 9, ndist_table)
    return thumb_index_nd < (mcp_ref_nd * 1.5)


def is_mrigasheersha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    pinky_straight = is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.85)
    thumb_straight = is_finger_straight(landmarks, dist_table, ndist_table, 2, 3, 4, 0.80)
    index_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, 0.85)
    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.85)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.85)
    if not (pinky_straight and thumb_straight and index_bent and middle_bent and ring_bent):
        return False
    v_thumb = (landmarks[4].x - landmarks[2].x, landmarks[4].y - landmarks[2].y)
    v_hand = (landmarks[9].x - landmarks[0].x, landmarks[9].y - landmarks[0].y)
    ang = angle_between(v_thumb, v_hand)
    if ang is None:
        return False
    return ang > 35


def is_simhamukha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    ref = norm_dist(5, 9, ndist_table)
    if ref == 0:
        return False

    if not is_finger_straight(landmarks, dist_table, ndist_table, 5, 6, 8, 0.90):
        return False
    if not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.90):
        return False

    d_tm = norm_dist(4, 12, ndist_table)
    d_tr = norm_dist(4, 16, ndist_table)
    d_mr = norm_dist(12, 16, ndist_table)
    cluster_thresh = ref * 1.9
    if not (d_tm < cluster_thresh and d_tr < cluster_thresh and d_mr < cluster_thresh):
        return False

    mid_straight = is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.92)
    ring_straight = is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.92)
    if mid_straight and landmarks[12].y < landmarks[9].y:
        return False
    if ring_straight and landmarks[16].y < landmarks[13].y:
        return False

    return True


def is_chatura_mudra(landmarks, distances, ndistances, scale_ref):
    if not (
        is_finger_straight(landmarks, distances, ndistances, 5, 6, 8, 0.85) and
        is_finger_straight(landmarks, distances, ndistances, 9, 10, 12, 0.85) and
        is_finger_straight(landmarks, distances, ndistances, 13, 14, 16, 0.85) and
        is_finger_straight(landmarks, distances, ndistances, 17, 18, 20, 0.85)
    ):
        return False

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]

    d_im = math.dist((index_tip.x, index_tip.y), (middle_tip.x, middle_tip.y))
    d_mr = math.dist((middle_tip.x, middle_tip.y), (ring_tip.x, ring_tip.y))

    if d_im > scale_ref * 0.30 or d_mr > scale_ref * 0.30:
        return False

    wrist = landmarks[0]
    middle_mcp = (landmarks[9].x, landmarks[9].y)
    palm_center = ((wrist.x + middle_mcp[0]) / 2, (wrist.y + middle_mcp[1]) / 2)

    thumb_tip = (landmarks[4].x, landmarks[4].y)
    index_mcp = (landmarks[5].x, landmarks[5].y)
    ring_mcp = (landmarks[13].x, landmarks[13].y)
    pinky_mcp = (landmarks[17].x, landmarks[17].y)

    d_thumb_palm = math.dist(thumb_tip, palm_center)
    d_thumb_index = math.dist(thumb_tip, (landmarks[8].x, landmarks[8].y))
    d_thumb_middle = math.dist(thumb_tip, (landmarks[12].x, landmarks[12].y))
    d_thumb_ring = math.dist(thumb_tip, (landmarks[16].x, landmarks[16].y))
    d_thumb_pinky = math.dist(thumb_tip, (landmarks[20].x, landmarks[20].y))

    tolerance = scale_ref * 0.01
    thumb_depth_ok = thumb_tip[1] > (middle_mcp[1] - tolerance)
    thumb_x_inside = min(index_mcp[0], pinky_mcp[0]) < thumb_tip[0] < max(index_mcp[0], pinky_mcp[0])
    thumb_deep_inside = (
        d_thumb_palm < d_thumb_index and
        d_thumb_palm < d_thumb_middle and
        d_thumb_palm < d_thumb_ring and
        d_thumb_palm < d_thumb_pinky
    )

    if not (thumb_depth_ok and thumb_deep_inside and thumb_x_inside):
        return False

    return True


def is_bhramara_mudra(landmarks, distances, ndistances, scale_ref):
    thumb_tip = (landmarks[4].x, landmarks[4].y)
    middle_tip = (landmarks[12].x, landmarks[12].y)
    d_tm = math.dist(thumb_tip, middle_tip)
    if d_tm > scale_ref * 0.22:
        return False

    ring_straight = is_finger_straight(landmarks, distances, ndistances, 13, 14, 16, 0.85)
    pinky_straight = is_finger_straight(landmarks, distances, ndistances, 17, 18, 20, 0.85)
    if not (ring_straight and pinky_straight):
        return False

    index_tip = (landmarks[8].x, landmarks[8].y)
    index_mcp = (landmarks[5].x, landmarks[5].y)
    d_index_fold = math.dist(index_tip, index_mcp)
    if d_index_fold > scale_ref * 0.22:
        return False

    return True


def is_hamsasya_mudra(landmarks, dist_table, ndist_table, scale_ref):
    
    # ---- 0) Prevent conflict with Arala ----
    try:
        if is_arala_mudra(landmarks, dist_table, ndist_table, scale_ref):
            return False
    except:
        pass
    
    # ---- 1) Thumb tip & Index tip must be VERY close ----
    d_thumb_index_tip = norm_dist(4, 8, ndist_table)
    if d_thumb_index_tip > 0.28 * scale_ref:
        return False

    # ---- 2) Index must be clearly bent ----
    index_is_straight = is_finger_straight(
        landmarks, dist_table, ndist_table,
        5, 6, 8, threshold=0.94
    )
    if index_is_straight:
        return False

    # ---- 3) Middle, Ring, Pinky must be straight ----
    middle_straight = is_finger_straight(landmarks, dist_table, ndist_table,
                                         9,10,12, threshold=0.93)
    ring_straight   = is_finger_straight(landmarks, dist_table, ndist_table,
                                         13,14,16, threshold=0.92)
    pinky_straight  = is_finger_straight(landmarks, dist_table, ndist_table,
                                         17,18,20, threshold=0.90)

    if not (middle_straight and ring_straight and pinky_straight):
        return False

    return True



def is_hamsapaksha_mudra(landmarks, dist_table, ndist_table, scale_ref):
    y_i = landmarks[8].y
    y_m = landmarks[12].y
    y_r = landmarks[16].y
    y_p = landmarks[20].y

    min_gap = scale_ref * 0.06
    if not (y_p + min_gap < min(y_r, y_m, y_i)):
        return False

    row_tol = scale_ref * 0.16
    if not (abs(y_i - y_m) < row_tol and
            abs(y_m - y_r) < row_tol and
            abs(y_i - y_r) < row_tol):
        return False

    thumb_i = norm_dist(4, 8, ndist_table)
    thumb_m = norm_dist(4, 12, ndist_table)
    if thumb_i < scale_ref * 0.25 or thumb_m < scale_ref * 0.25:
        return False

    return True


def is_mukula_mudra(landmarks, dist_table, ndist_table, scale_ref):
    tips = [4, 8, 12, 16, 20]
    pts = [(landmarks[t].x, landmarks[t].y) for t in tips]

    cx = sum(p[0] for p in pts) / 5
    cy = sum(p[1] for p in pts) / 5

    dists = [math.dist(p, (cx, cy)) for p in pts]
    if max(dists) > scale_ref * 3.6:
        return False

    close_count = 0
    for t in [8, 12, 16, 20]:
        if norm_dist(4, t, ndist_table) < scale_ref * 1.95:
            close_count += 1
    if close_count < 3:
        return False

    wrist = (landmarks[0].x, landmarks[0].y)
    mcp_ids = [2, 5, 9, 13, 17]
    for tip, mcp in zip(tips, mcp_ids):
        if math.dist(wrist, (landmarks[tip].x, landmarks[tip].y)) < \
           math.dist(wrist, (landmarks[mcp].x, landmarks[mcp].y)) + scale_ref * 0.05:
            return False

    return True


def is_tamrachuda_mudra(landmarks, dist_table, ndist_table, scale_ref):
    ref = nd(5, 9, ndist_table)
    if ref == 0:
        return False

    thumb_to_index_tip = nd(4, 8, ndist_table)
    if thumb_to_index_tip < ref * 0.75:
        return False

    pip_angle = get_angle(landmarks[5], landmarks[6], landmarks[7])
    dip_angle = get_angle(landmarks[6], landmarks[7], landmarks[8])
    pip_half = pip_angle > 120
    dip_bent = dip_angle < 155
    index_half_bent = pip_half and dip_bent
    if not index_half_bent:
        return False

    t_mid = nd(4, 10, ndist_table)
    t_mid2 = nd(4, 11, ndist_table)
    t_rip = nd(4, 14, ndist_table)
    t_rip2 = nd(4, 15, ndist_table)
    touch_thresh = ref * 1.45
    touches_mid_ring = (
        t_mid < touch_thresh or
        t_mid2 < touch_thresh or
        t_rip < touch_thresh or
        t_rip2 < touch_thresh
    )
    if not touches_mid_ring:
        return False

    middle_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 9, 10, 12, 0.90)
    ring_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 13, 14, 16, 0.90)
    pinky_bent = not is_finger_straight(landmarks, dist_table, ndist_table, 17, 18, 20, 0.90)
    if not (middle_bent and ring_bent and pinky_bent):
        return False

    return True


def is_alapadma_mudra(landmarks, dist_table, ndist_table, scale_ref):
    def curved(pip, dip, tip):
        ang = get_angle(landmarks[pip], landmarks[dip], landmarks[tip])
        return 110 < ang < 165

    curved_index = curved(6, 7, 8)
    curved_middle = curved(10, 11, 12)
    curved_ring = curved(14, 15, 16)
    curved_pinky = curved(18, 19, 20)
    if not (curved_index and curved_middle and curved_ring and curved_pinky):
        return False

    thumb_flex = get_angle(landmarks[2], landmarks[3], landmarks[4])
    if thumb_flex < 85:
        return False

    tips = [landmarks[i] for i in [8, 12, 16, 20]]
    ys = [t.y for t in tips]
    if max(ys) - min(ys) < 0.08:
        return False

    d_im = nd(8, 12, ndist_table)
    d_mr = nd(12, 16, ndist_table)
    d_rp = nd(16, 20, ndist_table)
    if not (d_im > scale_ref * 1.2 and d_mr > scale_ref * 1.2 and d_rp > scale_ref * 1.1):
        return False

    wrist = landmarks[0]
    tip_dists = [
        math.dist((wrist.x, wrist.y), (landmarks[8].x, landmarks[8].y)),
        math.dist((wrist.x, wrist.y), (landmarks[12].x, landmarks[12].y)),
        math.dist((wrist.x, wrist.y), (landmarks[16].x, landmarks[16].y)),
        math.dist((wrist.x, wrist.y), (landmarks[20].x, landmarks[20].y)),
    ]
    if min(tip_dists) < scale_ref * 1.2:
        return False

    return True


def is_kangulashya_mudra(landmarks, dist_table, ndist_table, scale_ref):
    ring_angle = get_angle(landmarks[13], landmarks[14], landmarks[16])
    ring_bent = ring_angle < 155
    if not ring_bent:
        return False

    def straight(pip, dip, tip):
        return is_finger_straight(landmarks, dist_table, ndist_table, pip, dip, tip, 0.85)

    thumb_straight = straight(2, 3, 4)
    index_straight = straight(5, 6, 8)
    middle_straight = straight(9, 10, 12)
    pinky_straight = straight(17, 18, 20)
    if not (thumb_straight and index_straight and middle_straight and pinky_straight):
        return False

    ring_tip_i = nd(16, 8, ndist_table)
    ring_tip_m = nd(16, 12, ndist_table)
    ring_tip_p = nd(16, 20, ndist_table)
    min_dist = min(ring_tip_i, ring_tip_m, ring_tip_p)
    if min_dist < scale_ref * 0.9:
        return False

    d_im = nd(8, 12, ndist_table)
    d_mp = nd(12, 20, ndist_table)
    d_ip = nd(8, 20, ndist_table)
    if not (d_im < scale_ref * 2.4 and d_mp < scale_ref * 2.6 and d_ip < scale_ref * 3.0):
        return False

    y_vals = [landmarks[i].y for i in [4, 8, 12, 20]]
    if max(y_vals) - min(y_vals) > 0.18:
        return False

    return True


# ============================================================
# Mudra registry (order = priority)
# ============================================================
mudra_functions = {
    # High priority (very distinctive)
    "Musthi Mudra": is_musthi_mudra,
    "Suchi Mudra": is_suchi_mudra,
    "Kapitha Mudra": is_kapitha_mudra,
    "Shikharam Mudra": is_shikharam_mudra,
    "Hamsasya Mudra": is_hamsasya_mudra,
    "Tripataka Mudra": is_tripataka_mudra,
    "Mayura Mudra": is_mayura_mudra,

    # Medium priority
    "Simhamukha Mudra": is_simhamukha_mudra,
    "Tamrachuda Mudra": is_tamrachuda_mudra,
    "Bhramara Mudra": is_bhramara_mudra,
    "Katakamukha Mudra": is_katakamukha_mudra,

    # Lower priority / more ambiguous
    "Arala Mudra": is_arala_mudra,
    "Mrigasheersha Mudra": is_mrigasheersha_mudra,
    "Sarpashirsha Mudra": is_sarpashirsha_mudra,
    "Ardhapataka Mudra": is_ardhapataka_mudra,

    # Wide / bowl shapes & special
    "Padmakosha Mudra": is_padmakosha_mudra,
    "Alapadma Mudra": is_alapadma_mudra,
    "Kangulashya Mudra": is_kangulashya_mudra,
    "Hamsapaksha Mudra": is_hamsapaksha_mudra,
    "Chatura Mudra": is_chatura_mudra,

    # Remaining classical shapes
    "Pataka Mudra": is_pataka_mudra,
    "Ardha Chandra Mudra": is_ardha_chandra_mudra,
    "Shuka Tundam Mudra": is_shuka_tundam_mudra,
    "Kartari Mukham Mudra": is_kartari_mukham_mudra,
    "Chandrakala Mudra": is_chandrakala_mudra,
    "Mukula Mudra": is_mukula_mudra,
    "Trishula Mudra": is_trishula_mudra,
}

# ============================================================
# Mediapipe setup
# ============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    """Main function to run the desktop OpenCV version"""
    global DEBUG
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            mudra_status = "No Mudra Detected"
            debug_text = ""
            text_color = (0, 0, 255)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                dist_table, ndist_table, scale_ref = compute_distance_tables(landmarks)

                for name, func in mudra_functions.items():
                    try:
                        detected = func(landmarks, dist_table, ndist_table, scale_ref)
                    except Exception as e:
                        detected = False
                        if DEBUG:
                            debug_text = f"{name} error: {str(e)[:30]}"
                    if detected:
                        mudra_status = f"{name} Detected"
                        text_color = (0, 255, 0)
                        if DEBUG:
                            debug_text = f"{name} | scale={scale_ref:.3f}"
                        break

            cv2.putText(
                frame,
                mudra_status,
                (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                text_color,
                3,
                cv2.LINE_AA,
            )

            if DEBUG and debug_text:
                cv2.putText(
                    frame,
                    debug_text,
                    (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Mudra Detector", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            if key == ord('d'):
                DEBUG = not DEBUG

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

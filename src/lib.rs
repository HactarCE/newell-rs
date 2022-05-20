//! To match egui coordinate system, +1 is clockwise and -1 is counterclockwise.

use cgmath::*;
use std::cmp::Ordering;

const EPSILON: f32 = 0.0001;

pub trait NewellObj: Sized {
    fn name(&self) -> &str;
    fn area(&self) -> f32;

    /// Aprroximates depth comparison. This method does not need to be accurate,
    /// but it should be fast.
    fn approx_depth_cmp(&self, other: &Self) -> Ordering;

    /// Returns `true` if `self` can be drawn behind `other`. Only returns
    /// `false` if `self` _must_ be drawn in front of `other`.
    fn can_be_drawn_behind(&self, other: &Self, log: &mut String) -> bool;

    /// Splits `self` into two objects, divided by the plane of `other`:
    /// `(behind, in_front)`. Returns `None` if `self` is completely in front of
    /// `other` or completely behind `other`.
    fn split_by(&self, other: &Self) -> Option<(Self, Self)>;
}

/// Sort objects by depth according to Newell's algorithm.
pub fn sort(objs: &mut Vec<impl NewellObj>, max_splits: usize, log: &mut String) -> usize {
    // First, approximate the correct order.
    objs.sort_by(NewellObj::approx_depth_cmp);

    let mut splits_done = 0;

    // This algorithm is basically selection sort. At every iteration, all the
    // objects before `i` are guaranteed to be in their final order, and we
    // search for the object that we can place in the `i`th index.
    let mut i = 0;
    while i < objs.len() {
        // Keep track of how many times we swap objects; if we swap objects too
        // many times, we'll need to split one of them.
        let mut swaps = 0;

        // We want to advance `i`. In order to do that, we need to know that
        // `objs[i]` can be drawn behind all of `objs[(i+1)..]`.
        let mut j = i + 1;
        while j < objs.len() {
            *log += &format!(
                "Order is now: {}\n",
                objs.iter().map(|o| o.name()).collect::<Vec<_>>().join(",")
            );
            *log += &format!("Checking whether {} < {}\n", objs[i].name(), objs[j].name());
            if objs[i].can_be_drawn_behind(&objs[j], log) {
                *log += &format!("... and it is!\n");
                // All good! Advance `j`.
                j += 1;
            } else if i + swaps > j {
                *log += &format!("... it is not.\n");
                let i_name = objs[i].name();
                let i_area = objs[i].area();
                *log += &format!(
                    "... Want to split something w.r.t. {i_name} 2 {i} (area={i_area:.02})\n"
                );
                if splits_done >= max_splits {
                    *log += "... ... but can't :(\n\n";
                    // Just give up. :(
                    break;
                }
                splits_done += 1;

                // Hey wait, we've already tried swapping this polygon! There
                // must be a cycle, which can only be resolved by splitting one
                // of the objects. At some point the cycle has to cross through
                // the plane of `objs[i]`, so try splitting one of the objects
                // by `objs[i]` and hope that helps.
                for k in (i + 1)..=j {
                    let k_name = objs[k].name();
                    let k_area = objs[k].area();
                    if let Some((behind, in_front)) = objs[k].split_by(&objs[i]) {
                        *log += &format!(
                            "... ... successfully split {k_name} @ {k} (area={k_area:.02})\n"
                        );
                        *log += &format!(
                            "... ... behind is {} (area={:.02})\n",
                            behind.name(),
                            behind.area()
                        );
                        *log += &format!(
                            "... ... in front is {} (area={})\n\n",
                            in_front.name(),
                            in_front.area()
                        );
                        objs[k] = behind;
                        objs.push(in_front);
                        break;
                    }
                }
                // We know everything up until `i` is still good, but we
                // need to check all of `objs[(i+1)..]` again.
                j = i + 1;
                // Reset the swap counter so that we don't immediately try to
                // split again.
                swaps = 0;
            } else {
                *log += &format!(
                    "... Rotating {} to before {}\n",
                    objs[j].name(),
                    objs[i].name()
                );
                // Uh oh, `objs[j]` must be drawn behind `objs[i]`. Select
                // `objs[j]` to be drawn next by putting it at index `i` and
                // shifting everything else to the right.
                objs[i..=j].rotate_right(1);
                // Record that we swapped this polygon.
                swaps += 1;
                // Check all of `objs[(i+1)..]` again.
                j = i + 1;
            }
        }

        // Now we know that `objs[i]` can be drawn behind all of
        // `objs[(i+1)..]`, so we can advance `i`.
        i += 1;
    }

    // Everything is sorted now! Yay!
    splits_done
}

pub struct Polygon {
    name: String,
    verts: Vec<Point3<f32>>,
    normal: Vector3<f32>,
    min_bound: Point3<f32>,
    max_bound: Point3<f32>,
}
impl Polygon {
    pub fn verts(&self) -> &[Point3<f32>] {
        &self.verts
    }

    /// Returns the height of a point above or below the plane of `self`.
    fn height_of_point(&self, point: Point3<f32>) -> f32 {
        (point - self.verts[0]).dot(self.normal)
    }

    /// Returns the screen-space intersection of `self` and `other`.
    fn xy_intersection(&self, other: &Self) -> Option<Self> {
        let mut verts = self.verts.clone();
        for (o1, o2) in other.edges() {
            let other_line = (point2(o1.x, o1.y), point2(o2.x, o2.y));

            let mut new_verts = vec![];

            for self_edge in verts
                .iter()
                .map(move |&v| PointRelativeToLine::new(v, other_line))
                .cyclic_pairs()
                .map(|(a, b)| LineSegmentRelativeToLine { a, b })
            {
                // TODO: clockwise or counterclockwise?
                if self_edge.a.h < -EPSILON {
                    new_verts.push(self_edge.a.p);
                }
                if let Some(intermediate) = self_edge.intersection() {
                    new_verts.push(intermediate);
                }
            }

            verts = new_verts;
        }

        Polygon::new(verts, format!("({} & {})", self.name, other.name))
    }

    fn edges<'a>(&'a self) -> impl 'a + Iterator<Item = (Point3<f32>, Point3<f32>)> {
        let v1s = self.verts.iter().copied();
        let v2s = self.verts.iter().copied().cycle().skip(1);
        v1s.zip(v2s)
    }
    fn edges_relative_to_plane<'a>(
        &'a self,
        plane: (Point3<f32>, Vector3<f32>),
    ) -> impl 'a + Iterator<Item = LineSegmentRelativeToPlane> {
        self.verts
            .iter()
            .map(move |&v| PointRelativeToPlane::new(v, plane))
            .cyclic_pairs()
            .map(|(a, b)| LineSegmentRelativeToPlane { a, b })
    }
}
impl NewellObj for Polygon {
    fn approx_depth_cmp(&self, other: &Self) -> Ordering {
        f32_total_cmp(&self.min_bound.z, &other.min_bound.z)
    }

    fn can_be_drawn_behind(&self, other: &Self, log: &mut String) -> bool {
        // 1. If `self` is completely behind `other`, then `self` can be drawn
        //    behind.
        if self.max_bound.z < other.min_bound.z {
            *log += "because of Z bound:\n";
            return true;
        }

        // 2. If the bounding boxes of `self` and `other` do not overlap on the
        //    screen, then `self` can be drawn behind.
        if self.max_bound.x < other.min_bound.x
            || self.max_bound.y < other.min_bound.y
            || other.max_bound.x < self.min_bound.x
            || other.max_bound.y < self.min_bound.y
        {
            *log += "because of XY bounding rect:\n";
            return true;
        }

        // 3. If every vertex of `self` is behind the plane of `other`, then
        //    `self` can be drawn behind.
        if self
            .verts
            .iter()
            .all(|&v| other.height_of_point(v) <= EPSILON)
        {
            *log += "because all points are behind plane:\n";
            return true;
        }

        // 4. If every vertex of `other` is in front of the plane of `self`,
        //    then `self` can be drawn behind.
        if other
            .verts
            .iter()
            .all(|&v| self.height_of_point(v) >= -EPSILON)
        {
            *log += "because all points are in front of plane:\n";
            return true;
        }

        // 5. If `self` and `other` do not overlap on the screen, then `self`
        //    can be drawn behind.
        if let Some(intersection) = self.xy_intersection(other) {
            // 6. If `self` is always behind the plane of `other` whenever they
            //    intersect, then `self` can be drawn behind.
            if intersection
                .verts
                .iter()
                .all(|&v| other.height_of_point(v) <= EPSILON)
            {
                *log += "because all of intersection is behind plane:\n";
                return true;
            } else {
                // If we've reached this point, then there is some part of
                // `self` that must be drawn in front of `other`.
                return false;
            }
        } else {
            *log += "because no 2D intersection:\n";
            return true;
        }
    }

    fn split_by(&self, other: &Self) -> Option<(Self, Self)> {
        let mut verts_in_front = vec![];
        let mut verts_behind = vec![];

        let self_edges = self.edges_relative_to_plane((other.verts[0], other.normal));

        // // Each dot product represents the distance of a point from `self` above
        // // or below the plane of `other`.
        // let dots = self
        //     .verts
        //     .iter()
        //     .map(|v| (v - other.verts[0]).dot(other.normal));

        // // Iterate over adjacent pairs of `(vertex, dot_product)`.
        // let verts_and_dots: Vec<(Point3<f32>, f32)> = self
        //     .verts
        //     .iter()
        //     .copied()
        //     .zip(dots)
        //     .cycle()
        //     .take(self.verts.len() + 1)
        //     .collect();

        for edge in self_edges {
            if edge.a.h > 0.0 {
                // `a` is in front.
                verts_in_front.push(edge.a.p);
            } else {
                // `a` is behind.
                verts_behind.push(edge.a.p);
            }
            if let Some(intermediate) = edge.intersection() {
                // `a` and `b` are on opposite sides; add an intermediate vertex
                // to both lists.
                verts_in_front.push(intermediate);
                verts_behind.push(intermediate);
            }
        }

        if verts_in_front.is_empty() || verts_behind.is_empty() {
            None
        } else {
            Some((
                Polygon::new(
                    verts_behind,
                    format!("({} behind {})", self.name(), other.name()),
                )?,
                Polygon::new(
                    verts_in_front,
                    format!("({} in front of {})", self.name(), other.name()),
                )?,
            ))
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn area(&self) -> f32 {
        let p1 = self.verts[0];
        let mut area = 0.0;
        for (p2, p3) in self.verts[1..].iter().zip(&self.verts[2..]) {
            area += (p3 - p1).cross(p2 - p1).magnitude();
        }
        area
    }
}
impl Polygon {
    /// Constructs a convex polygon from a list of coplanar vertices in
    /// counterclockwise order. Returns `None` if the polygon is degenerate, or
    /// if the first three vertices are colinear.
    pub fn new(verts: Vec<Point3<f32>>, name: String) -> Option<Self> {
        // If there are fewer than three vertices, the polygon is degenerate.
        if verts.len() < 3 {
            return None;
        }

        // If any vertices are nonfinite, then the polygon is degenerate.
        if verts.iter().any(|v| !v.is_finite()) {
            return None;
        }

        // If the first three vertices are colinear, then consider the polygon
        // degenerate.
        let normal = (verts[1] - verts[0]).cross(verts[2] - verts[0]);
        if normal.magnitude2().is_zero() {
            return None;
        }

        Some(Self::new_unchecked(verts, name))
    }

    /// Constructs a convex polygon from a list of coplanar vertices in
    /// counterclockwise order. The polygon must not be degenerate, and no three
    /// vertices may be colinear.
    pub fn new_unchecked(verts: Vec<Point3<f32>>, name: String) -> Self {
        let mut min_bound = verts[0];
        let mut max_bound = verts[0];
        for v in &verts[1..] {
            if v.x < min_bound.x {
                min_bound.x = v.x;
            }
            if v.y < min_bound.y {
                min_bound.y = v.y;
            }
            if v.z < min_bound.z {
                min_bound.z = v.z;
            }

            if v.x > max_bound.x {
                max_bound.x = v.x;
            }
            if v.y > max_bound.y {
                max_bound.y = v.y;
            }
            if v.z > max_bound.z {
                max_bound.z = v.z;
            }
        }

        let normal = (verts[1] - verts[0]).cross(verts[2] - verts[0]).normalize();

        Self {
            name,

            verts,
            normal,
            min_bound,
            max_bound,
        }
    }

    // /// Returns the index of the vertex of the polygon that is farthest along
    // /// `direction` in sceen-space.
    // fn support_point_index(&self, direction: Vector2<f32>) -> usize {
    //     // TODO: optimize to O(log n) time instead of O(n) time.
    //     self.verts
    //         .iter()
    //         .map(|v| v.to_vec().truncate().dot(direction))
    //         .enumerate()
    //         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    //         .map(|(i, _)| i)
    //         .unwrap_or(0)
    // }
    // /// Returns the point of the polygon that is farthest along `direction` in
    // /// sceen-space.
    // fn support_point(&self, direction: Vector2<f32>) -> Point3<f32> {
    //     self.verts[self.support_point_index(direction)]
    // }

    // /// Returns whether two polygons overlap in screen-space.
    // pub fn overlaps(&self, other: &Self) -> bool {
    //     if !self.bounding_box_overlaps(other) {
    //         return false;
    //     }

    //     // Use GJK algorithm to find intersecting point.
    //     let mut support_vector = Vector2::unit_x();
    //     let mut self_support_a = None;
    //     let mut self_support_b = None;
    //     let mut self_support_c = None;
    //     let mut other_support_a = None;
    //     let mut other_support_b = None;
    //     let mut other_support_c = None;
    //     loop {
    //         let mut new_self_support = self.support_point(support_vector);
    //         let mut new_other_support = other.support_point(-support_vector);
    //         // self_support_a
    //         // Find the support point on the Minkowski difference.
    //         let mut support_point = (self.support_point(support_vector)
    //             - other.support_point(-support_vector))
    //         .truncate();
    //         support_vector = -support_point;
    //         // TODO: is the support point past the origin?
    //     }
    // }

    // /// Returns whether two polygons' bounding boxes overlap in screen-space.
    // fn bounding_box_overlaps(&self, other: &Self) -> bool {
    //     !(self.max_bound.x < other.min_bound.x
    //         || self.max_bound.y < other.min_bound.y
    //         || self.max_bound.z < other.min_bound.z
    //         || other.max_bound.x < self.min_bound.x
    //         || other.max_bound.y < self.min_bound.y
    //         || other.max_bound.z < self.min_bound.z)
    // }

    // /// Returns the order of this polygon relative to `other`.
    // pub fn order(&self, other: &Self) -> PolygonOrdering {
    //     if !self.overlaps(other) {
    //         return PolygonOrdering::DoesntMatter;
    //     }

    //     if self.max_bound.z < other.min_bound.z {
    //         return PolygonOrdering::Behind;
    //     }
    //     if other.max_bound.z < self.min_bound.z {
    //         return PolygonOrdering::InFront;
    //     }

    //     todo!("polygon ordering")
    // }

    // /// Splits a polygon along the plane of another. Returns a tuple `(in_front, behind)`
    // pub fn split(&self, intersecting: &Self) -> (Option<Self>, Option<Self>) {}
}

// enum PolygonOrdering {
//     DoesntMatter,
//     InFront,
//     Behind,
//     Intersects,
// }

// /// Returns whether the triangle formed by three points contain the origin.
// fn triangle_contains_origin(a: Point2<f32>, b: Point2<f32>, c: Point2<f32>) -> bool {
//     // winding()
//     todo!()
// }

// /// Returns whether three points are clockwise, counterclockwise, or colinear.
// fn winding(a: Point2<f32>, b: Point2<f32>, c: Point2<f32>) -> Winding {
//     let det = Matrix3::from_cols(
//         a.to_vec().extend(1.0),
//         b.to_vec().extend(1.0),
//         c.to_vec().extend(1.0),
//     )
//     .determinant()
//     .signum();
//     if det.is_positive() {
//         Winding::Cw
//     } else if det.is_negative() {
//         Winding::Ccw
//     } else {
//         Winding::Colinear
//     }
// }

// #[repr(i8)]
// enum Winding {
//     Cw = 1,
//     Ccw = -1,
//     Colinear = 0,
// }

/// Stolen shamelessly from [`std::f32::total_cmp()`], which isn't stable yet at
/// the time of writing.
fn f32_total_cmp(a: &f32, b: &f32) -> Ordering {
    let mut left = a.to_bits() as i32;
    let mut right = b.to_bits() as i32;

    left ^= (((left >> 31) as u32) >> 1) as i32;
    right ^= (((right >> 31) as u32) >> 1) as i32;

    left.cmp(&right)
}

// /// Returns the intersection point along the segment from `a` to `b` given the
// /// dot product of `a` with the plane and `b` with the plane. The return value
// /// is between `a` and `b` iff `a_dot` and `b_dot` have opposite signs.
// fn intersect(a: Point3<f32>, b: Point3<f32>, a_dot: f32, b_dot: f32) -> Point3<f32> {}

// fn iter_points_and_heights_relative_to_plane(
//     points: impl Iterator<Item = Point3<f32>>,
//     plane: (Point3<f32>, Vector3<f32>),
// ) -> impl Iterator<Item = (Point3<f32>)> {
// }

/// Pair of points, each of which is some distance above or below a plane.
#[derive(Debug, Copy, Clone)]
struct LineSegmentRelativeToPlane {
    a: PointRelativeToPlane,
    b: PointRelativeToPlane,
}
impl LineSegmentRelativeToPlane {
    /// Returns the intersection of the line segment with the plane.
    fn intersection(self) -> Option<Point3<f32>> {
        (self.a.h.signum() != self.b.h.signum()).then(|| {
            let delta = self.b.p - self.a.p;
            let ratio = self.a.h / (self.a.h - self.b.h);
            self.a.p + delta * ratio
        })
    }
}

struct LineSegmentRelativeToLine {
    a: PointRelativeToLine,
    b: PointRelativeToLine,
}
impl LineSegmentRelativeToLine {
    /// Returns the intersection of the line segment with the line.
    fn intersection(self) -> Option<Point3<f32>> {
        (self.a.h.signum() != self.b.h.signum()).then(|| {
            let delta = self.b.p - self.a.p;
            let ratio = self.a.h / (self.a.h - self.b.h);
            self.a.p + delta * ratio
        })
    }
}

/// Point that is some distance above or below a plane.
#[derive(Debug, Copy, Clone)]
struct PointRelativeToPlane {
    /// Point.
    p: Point3<f32>,
    /// Height of point, relative to the plane.
    h: f32,
}
impl PointRelativeToPlane {
    fn new(p: Point3<f32>, plane: (Point3<f32>, Vector3<f32>)) -> Self {
        Self {
            p,
            h: (p - plane.0).dot(plane.1),
        }
    }
}

/// Point that is some distance above or below a plane.
#[derive(Debug, Copy, Clone)]
struct PointRelativeToLine {
    /// Point.
    p: Point3<f32>,
    /// Height of point, relative to the plane.
    h: f32,
}
impl PointRelativeToLine {
    fn new(p: Point3<f32>, line: (Point2<f32>, Point2<f32>)) -> Self {
        Self {
            p,
            h: (cgmath::point2(p.x, p.y) - line.0).perp_dot(line.1 - line.0),
        }
    }
}

trait IterCyclicPairsExt: Iterator + Sized {
    fn cyclic_pairs(self) -> CyclicPairsIter<Self>;
}
impl<I> IterCyclicPairsExt for I
where
    I: Iterator,
    I::Item: Clone,
{
    fn cyclic_pairs(mut self) -> CyclicPairsIter<Self> {
        let first = self.next();
        let prev = first.clone();
        CyclicPairsIter {
            first,
            prev,
            rest: self,
        }
    }
}

struct CyclicPairsIter<I: Iterator> {
    first: Option<I::Item>,
    prev: Option<I::Item>,
    rest: I,
}
impl<I> Iterator for CyclicPairsIter<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = (I::Item, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        Some(match self.rest.next() {
            Some(curr) => (self.prev.replace(curr.clone())?, curr),
            None => (self.prev.take()?, self.first.take()?),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.rest.size_hint();
        (lo.saturating_add(1), hi.and_then(|x| x.checked_add(1)))
    }
}

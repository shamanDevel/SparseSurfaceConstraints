#pragma once

#include <Eigen/Core>
#include <vector>
#include <cinder/CinderMath.h>

namespace ar {
namespace geom2d
{
	typedef long double decimal;
	typedef std::pair<decimal, decimal> point;
	struct line {
		decimal a, b, c;
		line() {}
		line(decimal a, decimal b, decimal c)
			: a(a), b(b), c(c) {}
		point normal() {
			return point(a, b);
		}
	};
	inline point middle(point const p1, point const p2) {
		return point((p1.first + p2.first) / 2.0, (p1.second + p2.second) / 2.0);
	}
	inline line lineByPoints(point const p1, point const p2) {
		decimal dx = p1.first - p2.first;
		decimal dy = p1.second - p2.second;
		return line(dy, -dx, -(dy*p1.first - dx * p1.second));
	}
	inline line lineByDirection(point const p, point const dir) {
		return line(dir.second, -dir.first, -(dir.second*p.first - dir.first*p.second));
	}

	inline bool isIdentical(line const l0, line const l1) {
		return
			(l0.a*l1.b == l0.b*l1.a) &&
			(l0.b*l1.c == l0.c*l1.b) &&
			(l0.a*l0.c == l0.c*l1.a);

	}
	inline bool isParallel(line const l0, line const l1) {
		return (l0.a*l1.b) == (l1.a*l0.b) != 0;
	}
	inline point intersection(line const l0, line const l1) {
		if (l0.a != 0) { //a0!=0
			decimal y = (l1.a / l0.a * l0.c - l1.c) / (l1.b - l1.a / l0.a * l0.b);
			decimal x = (l0.b * y + l0.c) / (-l0.a);
			return point(x, y);
		}
		else { //a0==0, assume b0!=0
			decimal x = (l1.b / l0.b * l0.c - l1.c) / (l1.a - l1.b / l0.b * l0.a);
			decimal y = (l0.a * x + l0.c) / (-l0.b);
			return point(x, y);
		}
	}
	inline bool between(decimal d1, decimal d2, decimal d) {
		if (d1 >= d2) {
			return (d1 >= d && d >= d2);
		}
		else {
			return (d2 >= d && d >= d1);
		}
	}
	inline bool between(point const p1, point const p2, point const p) { //checks if 'p' lies between p1 and p2
		return between(p1.first, p2.first, p.first)
			&& between(p1.second, p2.second, p.second);
	}
	inline decimal distance(line const l, point const p) {
		decimal d = (l.a*p.first + l.b*p.second + l.c) / sqrt(l.a*l.a + l.b*l.b);
		return abs(d);
	}
	inline decimal length(point const u, point const v) {
		decimal dx = u.first - v.first;
		decimal dy = u.second - v.second;
		return sqrt(dx*dx + dy * dy);
	}
	inline point rotate(point const p, decimal angleDegree) {
		decimal angle = angleDegree / 180.0 * M_PI;
		decimal s = sin(angle);
		decimal c = cos(angle);
		return point(p.first * c - p.second * s,

			p.first * s + p.second * c);

	}
	//Scale point p by a factor f around the center 'center'
	inline point scale(point const p, point const center, decimal f) {
		decimal dx = p.first - center.first;
		decimal dy = p.second - center.second;
		dx *= f;
		dy *= f;
		return point(dx, dy);
	}
	//Mirrors the point p at the specific line l
	inline point mirror(point const p, line const l) {
		line l2 = lineByDirection(p, point(l.a, l.b)); //Lot on l through p
		point cut = intersection(l, l2);
		return point(p.first + 2 * (cut.first - p.first),

			p.second + 2 * (cut.second - p.second));

	}
	//Calculates the areo of a (convex / non-convex) polygon
	inline decimal area(std::vector<point> const& points) {
		int n = points.size();
		decimal area = points[n - 1].first * points[0].second - points[n - 1].second * points[0].first;
		for (int i = 0; i<n - 1; ++i) {
			area += points[i].first * points[i + 1].second - points[i].second * points[i + 1].first;
		}
		area /= 2.0;
		return abs(area);
	}
}
}

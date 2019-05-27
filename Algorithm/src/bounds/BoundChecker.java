package bounds;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.Vector;

/*
 * 
 * Sadekujjaman Saju
 * 
 */

public class BoundChecker {

	public static void main(String[] args) {

		Scanner sc = new Scanner(System.in);

		// Vector<Integer> v = new Vector<>();
		//
		// v.add(4);
		// v.add(6);
		// v.add(8);
		// v.add(10);
		// v.add(12);
		// 1 4 6 8 10
		Integer v[] = new Integer[5];
		v[0] = 1;
		v[1] = 4;
		v[2] = 6;
		v[3] = 8;
		v[4] = 10;

		int l = lower_bound(v, 7);
		int u = upper_bound(v, 1000);
		if(u == -1){
			u = v.length;
		}
		System.out.println(l + " " + u);
		

		sc.close();
	}


	public static int lower_bound(Comparable[] arr, Comparable key) {
		int len = arr.length;
		int lo = 0;
		int hi = len - 1;
		int mid = (lo + hi) / 2;
		while (true) {
			int cmp = arr[mid].compareTo(key);
			if (cmp == 0 || cmp > 0) {
				hi = mid - 1;
				if (hi < lo)
					return mid;
			} else {
				lo = mid + 1;
				if (hi < lo)
					return mid < len - 1 ? mid + 1 : -1;
			}
			mid = (lo + hi) / 2;
		}
	}

	public static int upper_bound(Comparable[] arr, Comparable key) {
		int len = arr.length;
		int lo = 0;
		int hi = len - 1;
		int mid = (lo + hi) / 2;
		while (true) {
			int cmp = arr[mid].compareTo(key);
			if (cmp == 0 || cmp < 0) {
				lo = mid + 1;
				if (hi < lo)
					return mid < len - 1 ? mid + 1 : -1;
			} else {
				hi = mid - 1;
				if (hi < lo)
					return mid;
			}
			mid = (lo + hi) / 2;
		}
	}

}

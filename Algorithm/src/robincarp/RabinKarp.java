package robincarp;
import java.util.Scanner;

/**
 *
 * @author User
 */
public class RabinKarp {

    public static void main(String[] args) {
//        String text = "aaaabaaabaa";
//        String pattern = "aabaa";
//        String text = "aabaabcaaaba";
//        String pattern = "aab";

        Scanner sc = new Scanner(System.in);
        String text = sc.next();
        String pattern = sc.next();

        int n = text.length();
        int m = pattern.length();

        int mod = 997;

        long patternValue = 0;
        long previousSubStringValue = 0;
        for (int i = m - 1, j = 0; i >= 0; i--, j++) {
            int c = (int) pattern.charAt(j);
            patternValue = (patternValue + (c * pow(256, i, mod))) % mod;
            int d = (int) text.charAt(j);
            previousSubStringValue = (previousSubStringValue + (d * pow(256, i, mod))) % mod;
        }

//        System.out.println(patternValue + ", " + previousSubStringValue);
        if (previousSubStringValue == patternValue) {
            if (isEqual(text.substring(0, m), pattern)) {
                System.out.println("pattern found at point: " + 0);
            }
        }
        for (int i = m - 1 + 1; i < n; i++) {
            int previous = (int) text.charAt(i - m);
            int current = (int) text.charAt(i);
//            System.out.println("P " + previous + " " + current);

            long currentSubStringValue = (long) (previousSubStringValue - ((previous * pow(256, m - 1, mod)) % mod));
            if (currentSubStringValue < 0) {
                currentSubStringValue += mod;
            }
            currentSubStringValue = (current + (currentSubStringValue * 256)) % mod;

//            System.out.println("c: " + currentSubStringValue);
            if (currentSubStringValue == patternValue) {
                if (isEqual(text.substring(i - m + 1, i + 1), pattern)) {
                    System.out.println("Pattern Found at point: " + (i - m + 1));
                }

            }

            previousSubStringValue = currentSubStringValue;

        }
        sc.close();
    }

    private static boolean isEqual(String substring, String pattern) {

        for (int i = 0; i < pattern.length(); i++) {
            if (substring.charAt(i) != pattern.charAt(i)) {
                return false;
            }
        }
        return true;
    }

    private static long pow(int base, int pow, int mod) {
        if (pow == 0) {
            return 1;
        }
        if (pow % 2 == 0) {
            return pow((base * base) % mod, pow / 2, mod);
        } else {
            return (base * pow((base * base) % mod, (pow - 1) / 2, mod)) % mod;
        }
    }
}

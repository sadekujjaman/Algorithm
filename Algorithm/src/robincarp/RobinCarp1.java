package robincarp;

public class RobinCarp1 {

	public static void main(String[] args) {
		
	}
	static char text[] = new char[1000005];
	static char pattern[] = new char[1000005];
	static int d = 256, q = 11, h, m, p;
    static int counter = 0;
	static void find_h()
	{
	    h = 1;

	    for(int i = 1; i < m; ++i)
	        h = (h * d) % q;

	   // printf("The value of h is: %d\n", h);
	}

	static void pattern_hash_value()
	{
	    p = 0;

	    for(int i = 0; i < m; ++i)
	        p = (p * d + pattern[i]) % q;

	    //printf("The hash value of pattern is: %d\n", p);
	}

	static void rabin_karp_algo()
	{
	    int l = text.length, j, t = 0;

	    for(int i = 0; i < l; ++i)
	    {
	        if(i <= m - 1)
	            t = (t * d + text[i]) % q;
	        else
	        {
	            t = (d * (t - text[i - m] * h) + text[i]) % q;
	            if(t < 0)
	                t += q;
	        }

	       // printf("The value of t at index: %d is --> %d\n", i, t);
	        if((i >= m - 1) && (t == p))
	        {
	            for(j = 0; j < m; ++j)
	            {
	                if(pattern[j] != text[j + i - m + 1])
	                    break;
	            }

	            if(j == m){
	                counter++;
	                //printf("The pattern is found at index --> %d\n", i - m + 1);
	            }

	        }
	    }
	}

}

ORIFLAME Hackaton dataset description
==============

The database is available at https://dbhackaton.blob.core.windows.net/dbbackup/hackaton_db.bacpac 

You can download and import it to you local SQL Server, or you can ask for Azure subscription and Azure SQL.

The database contains several tables with information about Oriflame customers,
their orders, customer behavior on website, information about products, prices and stock.

There are 3 countries included - Turkey, Mexico and India. Note that some data were truncated or replaced. Some were even artificially created. 



CATALOGUE_PRICES
--------------
The table holds the data about prices pro product in specific time.

* COUNTRY_CODE - country code (IN-India, TR-Turkey, MX-Mexico)
* PROD_CD - product code (product identification)
* START_DATE - start of price validity
* END_DATE - end of price validity
* LOCAL_CURRENCY_PRICE - the price itself in local currency 


CUSTOMERS
-------------
The table holds sample of customers (so called consultants in Oriflame terminology) from 3 countries - Mexico, Turkey, India

* COUNTRY_CODE - country code (IN-India, TR-Turkey, MX-Mexico)
* CITY - City of the customer
* CUSTOMER_TYPE_ID - can have values
  * 1 - Standard consultant
  * 2 - VIP Customer. Consultant without contract with Oriflame. Consultant can't get bonuses.
  * 4 - SPO Owner. 'Company to company' model - Consultant make contract with Oriflame as company and set up SPO service center and get bonus for his sales
  * 103 - End customer with Oriflame account who can log-in to the Oriflame site. End customers are included into bonus tree, but doesn't receives bonus.
  * 201 - Oriflame Employee
* GENDER - Consultant's gender - Male/Female/Other
* BIRTH_DATE - consultants birthdate
* SIGNUP_DATE - the date when customer joined Oriflame
* CUSTOMER_STATUS - bit map with different statuses of customers
   * Activated = 0
   * Not Activated = 1 - only for consultant category - initial state of consultant after registration      
   * Terminated = 64 - terminated consultant who is excluded from bonus tree and can't login
   * Entrepreneur = 128
   * Vat Payer = 256
   * Retail = 512
   * Guarantee = 1024 - consultant who can make an order for his dowline
   * Credit Guarantee = 2048 - consultant who can guarantee credit for consultants from his downline
   * Reseller = 4096  - If customer resells the products (filled during the terms and conditions acceptance) 
   * Anonymized = 8192 - Anonymized customers, which haven't activated account during 90days period
   * Temporary = 16384 - Temporary created EndCustomer - created in shopping process - but not finished order.
   * BlockedDiscount = 32768 - can't get discount
* DISCOUNT - amount in local currency which can be used for discount for orders 
* CONSULTANT_NUMBER_S - unique identification of customer in specific country (anonymized for hackathon purposes) - used publicly 
* CUSTOMER_ID_S - unique identification of customer in specific country (anonymized for hackathon purposes) - internal identification
* RECRUIT_DATE - Date when consultant created first order with at least one business point (BP)
* TERMINATION_DATE - date when the consultant stopped cooperation with Oriflame. (null if consultant is active)
* TERMINATION_TYPE  
    * Not Terminated = 0
    * Automatic = 1 - Customer is terminated automatically by system
    * Manual = 2 - Customer is terminated manually
    * Manual Can't Rejoin = 3 - Customer is terminated manually and can't rejoin again into Oriflame (cheating, violation of Oriflame rules ...)


GA_EVENTS
----------------------
The table stores data from user tracking system on the website
* RECORD_DATE - date when the record was taken (without time)
* COUNTRY_CODE - country code (IN-India, TR-Turkey, MX-Mexico) of the site
* EVENT_CATEGORY - category of the event
* EVENT_ACTION - detail of the event
* EVENT_LABEL - used for multiple purposes - can be product code when action add to basket
* DEVICE_CATEGORY - user device which was used to trigger the event
* USER_AGENT - distinguish between Oriflame apps
* TOTAL_EVENTS - total ocurrence of the event - e.g. add to basket added 3 items then the value is 3
* CONSULTANT_NUMBER_S - consultant number of customer who triggered the event
* PAGE_PATH_S - page path where the event was collected (some number were removed from the path for security reasons)


GA_STATS
----------------------
The table stores data from user tracking system on the website
* RECORD_DATE - date when the record was taken (without time)
* COUNTRY_CODE - country code (IN-India, TR-Turkey, MX-Mexico) of the site
* COUNTRY - country where the request come from
* CITY - city where the request come from
* UNIQUE_PAGEVIEWS - number of page views during day and session
* PAGEVIEWS - number of the page views during whole day 
* CONSULTANT_NUMBER_S - consultant number of customer who triggered the event
* PAGE_PATH_S - page path where the event was collected (some number were removed from the path for security reasons)

MARKET_STOCK
-----------------------
Holds the information about stock of product in specific day
* DAY 
* COUNTRY_CODE - country code (IN-India, TR-Turkey, MX-Mexico)
* PROD_CD - product code
* SUM_QTY - quantity of the product in warehouse that day

ORDERS
-----------------------
Table ORDERS holds the information about orders 
* COUNTRY_CODE - country code (IN-India, TR-Turkey, MX-Mexico)
* ORDER_ID - unique identification of order in given country
* PERIOD_ID - period when the product was sold. Periods are usually 3weeks to 1 month long. 
* ORDER_DATE - date when the order packed and handed over to courier 
* ESTIMATED_DELIVERY_DATE - date when Oriflame estimates that the order can be delivered to customer
* DUE_DATE - date when the order has to be paid to
* CURRENCY_CODE - currency of the order
* ORDER_TYPE
    * N - Normal Order
    * S - Supplementary - Supplementary order made by Oriflame operator
* END_CUSTOMER_TYPE  
    * 0 - consultant
    * 3 - End customer without any sponsor
* ORIGIN 
    * I - Order created by Oriflame opearator in Oriflame immediate centre
    * O - Customer makes an order on Oriflame web site   
    * R - Order created during registration process
    * T - Order created by Oriflame opearator
    * V - Guarantee/leader makes an order for a customer from his downline
    * -- other
* STATUS - bitmap
    *  1 - Waiting for payment
    *  16 - Finalized
    *  64 - Canceled
* RELATED_ORDER_ID - when order is cancelled - new order (claim) is created with relation to previous one   
* SALES_TIME - data when the order was placed
* CLIENT_APP - the app of the client 
    * Online: 1
    * OriflameApp: 2 (Mobile app)
    * SkinCareApp: 4 (Mobile app)
    * Orisales: 99 (Legacy system)
* CUSTOMER_ID_S - unique identification of customer in specific country (anonymized for hackathon purposes) - internal identification


ORDER_ITEMS
-----------------------
Each order has list of order items
* COUNTRY_CODE - country code (IN-India, TR-Turkey, MX-Mexico)
* ORDER_ID - unique identification of order in given country
* ITEM_TYPE
   * Product: P - product added by customer (e.g. using quick entry form or from PDP)
   * Back order: O - dropped product - old demand
   * Bundle: B - bundle - item with dynamic structure
   * End customer order: C - dropped from end customer order
   * Service: S - dropped product
   * Autoship service: A - subscribed product
   * Merchandasing gift: G - product added by merchandising rule
   * Merchandasing change: H - product modified by merchandising rule (applied to item type P only, other item types are not touched)
   * Offer: M - product added by customer through offers
   * Replacement: L - replacement of out of stock product
   * Section: 1, 2, 3, 4 - products added from section
   * Free sample: E - sample product for free
   * Donation: V - donation fee
   * ​Fee: F - shipping or payment fee
   * Payment fee: J - e.g. card payment change
   * Registration fee: Z - change for new registered consultants
   * Renewal fee: R 
   * Extra sales: X
   * Free Shipping Joker​: ​Q  
   * Next order delivery: N - ​Dropping automatically added to the order
   * Reserved: W - ​Dropped product, originally reserved as Waiting order or Calendar campaign
   * Pre-order target: T
* ITEM_CODE - product code
* REQUESTED_QUANTITY - number of requested product (might be higher than ORDER_QUANTITY)
* ORDER_QUANTITY - number of products which are ordered
* RESERVATION_REASON 
   * On stock: O
   * Out of stock: U
   * Can not be sold: C
   * Refused backorder: R
   * Sales limit overflow: L
   * Extras sales overflow: E
   * Bundle component is not on the stock: B
* ORDER_ITEM_ID - unique identification of order item
* RELATED_ORDER_ITEM_ID - id of order item which the item is related to


PRODUCT_REVIEW
-----------------------
Consultant reviews of Oriflame products which are collected from website
* CREATION_DATE - when the comment was created by the customer
* PROD_CD - product code of product which the comment is related to
* COUNTRY_CODE - country code
* TEST - the comment of customer to the product
* RATING - Range 1 - 5 (5 is the best)
* DELETED - the comment was removed from the website
* REVIEW_USEFUL - number of customer who considered the review useful
* REVIEW_USELESS - number of customer who considered the review useless


PRODUCTS
-----------------------
All the products
* PROC_CD - product code
* PROC_DESCR - description of product
* COUNTRY_CODE - country code (IN-India, TR-Turkey, MX-Mexico)
* PROD_STATUS_DESCR - status of the product
* NPD_LAUNCH - 
* LONG_DESCR - long description of product
* CATEGORY_DESCR - category of the product
* DEVELOPMENT_DESCR - taxonomy segmentation
* SECTOR_DESCR - taxonomy segmentation
* SEGMENT_DESCR - taxonomy segmentation
* BRAND_DESCR - taxonomy segmentation
* SUBBRAND_DESCR - taxonomy segmentation
* TYPE_DESCR - taxonomy segmentation
* SET_SAMPLE_CD
* TEAM_CATEGORY_DESC - taxonomy segmentation
* PRICE_SEGMENT_DESC- taxonomy segmentation


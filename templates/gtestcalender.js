import 'dotenv/config'
import Nylas from 'nylas'


const NylasConfig = {
  apiKey: process.env.NYLAS_API_KEY,
  apiUri: process.env.NYLAS_API_URI,
}

const nylas = new Nylas(NylasConfig);

async function fetchFiveAvailableCalendars() {
  try {
    const calendars = await nylas.calendars.list({
      identifier: process.env.NYLAS_GRANT_ID,
      limit: 5
  })

  console.log('Available Calendars:', calendars);
  } catch (error) {
    console.error('Error fetching calendars:', error)
  }
}

fetchFiveAvailableCalendars()   